import math
import re
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from . import rssm
from .rssm import LinearLayer, get_act, get_norm, symexp, symlog


f32 = torch.float32
i32 = torch.int32


def sg(xs, skip: bool = False):
    """Stop gradient for tensors or container trees."""
    if skip:
        return xs
    if isinstance(xs, torch.Tensor):
        return xs.detach()
    if isinstance(xs, dict):
        return {k: sg(v) for k, v in xs.items()}
    if isinstance(xs, (list, tuple)):
        return type(xs)(sg(v) for v in xs)
    return xs


def prefix(xs: dict, p: str) -> dict:
    return {f"{p}/{k}": v for k, v in xs.items()}


def concat_tree(xs_list, dim):
    keys = xs_list[0].keys()
    return {k: torch.cat([x[k] for x in xs_list], dim) for k in keys}


def isimage(space):
    return hasattr(space, "dtype") and space.dtype == np.uint8 and len(space.shape) == 3


def tree_map(fn, *trees):
    if isinstance(trees[0], dict):
        return {k: tree_map(fn, *[t[k] for t in trees]) for k in trees[0].keys()}
    if isinstance(trees[0], (list, tuple)):
        return type(trees[0])(tree_map(fn, *[t[i] for t in trees]) for i in range(len(trees[0])))
    return fn(*trees)


class DistOutput:
    """Minimal distribution wrapper used by the various prediction heads."""

    def __init__(
        self,
        dist_type: str,
        params: torch.Tensor,
        space=None,
        *,
        bins: int = 255,
        minstd: float = 0.1,
        maxstd: float = 1.0,
        unimix: float = 0.0,
    ):
        self._dist_type = dist_type
        self._params = params
        self._space = space
        self._bins = bins
        self._minstd = minstd
        self._maxstd = maxstd
        self._unimix = unimix

    def pred(self) -> torch.Tensor:
        if self._dist_type == "mse":
            return self._params
        if self._dist_type == "binary":
            return torch.sigmoid(self._params)
        if self._dist_type == "symlog":
            return symexp(self._params)
        if self._dist_type == "twohot":
            return self._twohot_mean()
        if self._dist_type in ("onehot", "softmax", "categorical"):
            return F.softmax(self._categorical_logits(), dim=-1)
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            return self.mode()
        return self._params

    def prob(self, value) -> torch.Tensor:
        p = torch.sigmoid(self._params)
        return p if value == 1 else 1 - p

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        if self._dist_type == "mse":
            return 0.5 * (self._params - target).square().sum(-1)
        if self._dist_type == "binary":
            return F.binary_cross_entropy_with_logits(
                self._params, target.float(), reduction="none"
            ).sum(-1)
        if self._dist_type == "symlog":
            target_sg = symlog(target)
            if target_sg.dim() == self._params.dim() - 1:
                target_sg = target_sg.unsqueeze(-1)
            return 0.5 * (self._params - target_sg).square().sum(-1)
        if self._dist_type == "twohot":
            return self._twohot_loss(target)
        if self._dist_type in ("onehot", "softmax", "categorical"):
            logits = self._categorical_logits()
            target_idx = self._categorical_target(target)
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_idx.reshape(-1).long(),
                reduction="none",
            ).reshape(target_idx.shape)
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            return -self.logp(target)
        return 0.5 * (self._params - target).square().sum(-1)

    def logp(self, target: torch.Tensor) -> torch.Tensor:
        if self._dist_type in ("onehot", "softmax", "categorical"):
            logits = self._categorical_logits()
            target_idx = self._categorical_target(target)
            return -F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_idx.reshape(-1).long(),
                reduction="none",
            ).reshape(target_idx.shape)
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            dist = self._squashed_normal_dist()
            target = target.clamp(-0.999999, 0.999999)
            pre_tanh = torch.atanh(target)
            log_det = torch.log(1 - target.square() + 1e-6)
            return dist.log_prob(pre_tanh).sum(-1) - log_det.sum(-1)
        return -self.loss(target)

    def entropy(self) -> torch.Tensor:
        if self._dist_type in ("onehot", "softmax", "categorical"):
            p = F.softmax(self._categorical_logits(), dim=-1)
            return -(p * torch.log(p + 1e-8)).sum(-1)
        if self._dist_type == "binary":
            p = torch.sigmoid(self._params)
            return -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).sum(-1)
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            mean, std = self._normal_params()
            base = td.Normal(mean, std)
            pre_tanh = base.rsample()
            action = torch.tanh(pre_tanh)
            return -self.logp(action)
        return torch.zeros(self._params.shape[:-1], device=self._params.device)

    def sample(self) -> torch.Tensor:
        if self._dist_type in ("onehot", "softmax", "categorical"):
            dist = td.OneHotCategorical(logits=self._categorical_logits())
            sample = dist.sample()
            probs = F.softmax(self._categorical_logits(), dim=-1)
            return sample + probs - probs.detach()
        if self._dist_type == "binary":
            p = torch.sigmoid(self._params)
            sample = torch.bernoulli(p)
            return sample + p - p.detach()
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            pre_tanh = self._squashed_normal_dist().rsample()
            action = torch.tanh(pre_tanh)
            mean = self.mode()
            return action + mean - mean.detach()
        return self.pred()

    def mode(self) -> torch.Tensor:
        if self._dist_type in ("onehot", "softmax", "categorical"):
            logits = self._categorical_logits()
            index = logits.argmax(dim=-1)
            return F.one_hot(index, logits.shape[-1]).float()
        if self._dist_type == "binary":
            return (torch.sigmoid(self._params) >= 0.5).float()
        if self._dist_type == "twohot":
            return self.pred()
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            mean, _ = self._normal_params()
            return torch.tanh(mean)
        return self.pred()

    @property
    def minent(self):
        return 0.0

    @property
    def maxent(self):
        if self._dist_type in ("onehot", "softmax", "categorical"):
            return math.log(self._categorical_logits().shape[-1])
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            action_dim = self._normal_params()[0].shape[-1]
            return float(action_dim * 0.5 * math.log(2 * math.pi * math.e * (self._maxstd ** 2)))
        return 1.0

    def _twohot_mean(self) -> torch.Tensor:
        bins = self._twohot_bins()
        probs = F.softmax(self._params, dim=-1)
        return symexp((probs * bins).sum(-1))

    def _twohot_loss(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() > 0 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        bins = self._twohot_bins()
        target_sym = symlog(target)
        below = (bins <= target_sym.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, bins.shape[0] - 2)
        above = below + 1
        equal = (below == above).float()
        dist_to_below = torch.where(
            equal.bool(),
            torch.ones_like(equal),
            torch.abs(target_sym - bins[below]) / (bins[above] - bins[below] + 1e-8),
        )
        target_oh = F.one_hot(below, len(bins)) * (1 - dist_to_below.unsqueeze(-1))
        target_oh = target_oh + F.one_hot(above, len(bins)) * dist_to_below.unsqueeze(-1)
        log_probs = F.log_softmax(self._params, dim=-1)
        return -(target_oh.float() * log_probs).sum(-1)

    def _twohot_bins(self) -> torch.Tensor:
        return torch.linspace(-20, 20, self._bins, device=self._params.device)

    def _categorical_logits(self) -> torch.Tensor:
        if self._unimix <= 0:
            return self._params
        probs = F.softmax(self._params, dim=-1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        return torch.log(probs + 1e-8)

    def _categorical_target(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() > 0 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if target.shape[-1:] == self._params.shape[-1:]:
            return target.argmax(dim=-1)
        return target.long()

    def _normal_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_param = torch.chunk(self._params, 2, dim=-1)
        std = torch.sigmoid(std_param)
        std = self._minstd + (self._maxstd - self._minstd) * std
        return mean, std

    def _squashed_normal_dist(self) -> td.Normal:
        mean, std = self._normal_params()
        return td.Normal(mean, std)

    def stats(self) -> dict[str, torch.Tensor]:
        if self._dist_type in ("trunc_normal", "bounded_normal"):
            mean, std = self._normal_params()
            return {
                "mean": torch.tanh(mean),
                "std": std,
                "pre_tanh_mean": mean,
            }
        if self._dist_type in ("onehot", "softmax", "categorical"):
            probs = F.softmax(self._categorical_logits(), dim=-1)
            return {"prob": probs}
        return {"pred": self.pred()}


class MLPHead(nn.Module):
    """Shared MLP torso with light-weight distribution heads."""

    def __init__(
        self,
        out_space,
        dist_type=None,
        layers: int = 2,
        units: int = 1024,
        norm: str = "rms",
        act: str = "gelu",
        outscale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self._out_space = out_space
        self._layers = layers
        self._units = units
        if isinstance(dist_type, dict):
            self._dist_type = dist_type
        elif dist_type is not None:
            self._dist_type = dist_type
        elif isinstance(out_space, dict):
            self._dist_type = {
                k: ("onehot" if getattr(s, "discrete", False) else "mse") for k, s in out_space.items()
            }
        else:
            dtype = getattr(out_space, "dtype", np.float32)
            self._dist_type = "binary" if dtype in (bool, np.bool_) else "symlog"
        self._built = False
        self._norm = norm
        self._act = act
        self._outscale = outscale
        self._bins = kwargs.get("bins", 255)
        self._minstd = kwargs.get("minstd", 0.1)
        self._maxstd = kwargs.get("maxstd", 1.0)
        self._unimix = kwargs.get("unimix", 0.0)
        self.mlp = None
        self.heads = nn.ModuleDict()

    def _build(self, in_dim: int, device):
        mods = []
        d = in_dim
        for _ in range(self._layers):
            mods.append(LinearLayer(d, self._units))
            mods.append(get_norm(self._norm, self._units))
            mods.append(get_act(self._act)())
            d = self._units
        self.mlp = nn.Sequential(*mods).to(device)
        if isinstance(self._out_space, dict):
            for k, s in self._out_space.items():
                dt = self._canonical_dist_type(
                    self._dist_type[k] if isinstance(self._dist_type, dict) else self._dist_type
                )
                out_dim = self._output_dim(s, dt)
                self.heads[k] = LinearLayer(d, out_dim, self._outscale).to(device)
        else:
            dt = self._canonical_dist_type(self._dist_type if isinstance(self._dist_type, str) else "mse")
            out_dim = self._output_dim(self._out_space, dt)
            self.heads["_scalar"] = LinearLayer(d, out_dim, self._outscale).to(device)
        self._built = True

    def forward(self, inp: torch.Tensor, bdims: int = 2) -> Union[DistOutput, Dict[str, DistOutput]]:
        if not self._built:
            self._build(inp.shape[-1], inp.device)
        shape = inp.shape[:bdims]
        x = inp.reshape(-1, inp.shape[-1])
        x = self.mlp(x)
        if isinstance(self._out_space, dict):
            result = {}
            for k in self._out_space:
                out = self.heads[k](x).reshape(*shape, -1)
                dt = self._dist_type[k] if isinstance(self._dist_type, dict) else self._dist_type
                result[k] = DistOutput(
                    self._canonical_dist_type(dt),
                    out,
                    self._out_space[k],
                    bins=self._bins,
                    minstd=self._minstd,
                    maxstd=self._maxstd,
                    unimix=self._unimix,
                )
            return result
        out = self.heads["_scalar"](x).reshape(*shape, -1)
        dt = self._dist_type if isinstance(self._dist_type, str) else "mse"
        return DistOutput(
            self._canonical_dist_type(dt),
            out,
            self._out_space,
            bins=self._bins,
            minstd=self._minstd,
            maxstd=self._maxstd,
            unimix=self._unimix,
        )

    def _canonical_dist_type(self, dist_type: str) -> str:
        mapping = {
            "categorical": "onehot",
            "onehot": "onehot",
            "softmax": "onehot",
            "binary": "binary",
            "mse": "mse",
            "symlog": "symlog",
            "twohot": "twohot",
            "symexp_twohot": "twohot",
            "trunc_normal": "bounded_normal",
            "bounded_normal": "bounded_normal",
        }
        return mapping.get(dist_type, dist_type)

    def _output_dim(self, space, dist_type: str) -> int:
        shape = getattr(space, "shape", ())
        base_dim = max(int(np.prod(shape)), 1)
        dtype = getattr(space, "dtype", np.float32)
        if dist_type == "binary" or dtype in (bool, np.bool_):
            return 1
        if dist_type == "twohot":
            return self._bins
        if dist_type == "bounded_normal":
            return 2 * base_dim
        if getattr(space, "discrete", False):
            return shape[0] if len(shape) > 0 else 2
        return base_dim


class SlowModel(nn.Module):
    """EMA target network."""

    def __init__(self, target_module: nn.Module, source: nn.Module, decay: float = 0.98, update_every: int = 1, **kwargs):
        super().__init__()
        self.target = target_module
        self.source = source
        self.decay = decay
        self.update_every = update_every
        self._step = 0
        self._sync()

    def _sync(self):
        for tp, sp in zip(self.target.parameters(), self.source.parameters()):
            tp.data.copy_(sp.data)

    def update(self):
        self._step += 1
        if self._step % self.update_every == 0:
            with torch.no_grad():
                for tp, sp in zip(self.target.parameters(), self.source.parameters()):
                    tp.data.mul_(self.decay).add_(sp.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.target(*args, **kwargs)


class RunningNorm(nn.Module):
    """Running percentile-based normalization."""

    def __init__(self, decay: float = 0.99, limit: float = 1e-8, perclo: float = 5.0, perchi: float = 95.0, **kwargs):
        super().__init__()
        self.decay = decay
        self.limit = limit
        self.perclo = perclo
        self.perchi = perchi
        self.register_buffer("offset", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("_count", torch.tensor(0))

    def forward(self, x: torch.Tensor, update: bool = True):
        if update and x.numel() > 0:
            lo = torch.quantile(x.float().detach(), self.perclo / 100)
            hi = torch.quantile(x.float().detach(), self.perchi / 100)
            new_offset = (lo + hi) / 2
            new_scale = torch.clamp(hi - lo, min=self.limit)
            if self._count == 0:
                self.offset.copy_(new_offset)
                self.scale.copy_(new_scale)
            else:
                self.offset.mul_(self.decay).add_(new_offset, alpha=1 - self.decay)
                self.scale.mul_(self.decay).add_(new_scale, alpha=1 - self.decay)
            self._count.add_(1)
        return self.offset, self.scale

    def stats(self):
        return self.offset, self.scale


class DreamerOptimizer:
    """Near-official optimizer semantics: AGC -> RMS -> Momentum -> WD -> LR."""

    def __init__(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        *,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r"/kernel$",
        schedule: str = "const",
        warmup: int = 1000,
        anneal: int = 0,
    ):
        self.params = [(n, p) for (n, p) in named_params if p.requires_grad]
        self.lr = float(lr)
        self.agc = float(agc)
        self.eps = float(eps)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.use_momentum = bool(momentum)
        self.nesterov = bool(nesterov)
        self.wd = float(wd)
        self.wd_pattern = re.compile(wdregex)
        self.schedule = str(schedule)
        self.warmup = int(warmup)
        self.anneal = int(anneal)
        self.step_count = 0
        self.state: dict[str, dict[str, torch.Tensor]] = {}

    def zero_grad(self):
        for _, p in self.params:
            if p.grad is not None:
                p.grad = None

    def _lr_at(self, step: int) -> float:
        lr = self.lr
        if self.warmup > 0 and step <= self.warmup:
            lr *= step / max(1, self.warmup)
        if self.schedule == "linear" and self.anneal > 0:
            progress = min(max(step - self.warmup, 0) / max(1, self.anneal - self.warmup), 1.0)
            lr *= 1.0 - 0.9 * progress
        elif self.schedule == "cosine" and self.anneal > 0:
            progress = min(max(step - self.warmup, 0) / max(1, self.anneal - self.warmup), 1.0)
            lr *= 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(lr)

    def step(self) -> float:
        self.step_count += 1
        lr = self._lr_at(self.step_count)
        for name, p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()

            if self.agc > 0:
                p_norm = p.data.detach().float().norm(2).clamp(min=1e-3)
                g_norm = g.norm(2)
                max_norm = p_norm * self.agc
                if g_norm > max_norm:
                    g = g * (max_norm / (g_norm + 1e-8))

            st = self.state.setdefault(name, {})
            if "rms" not in st:
                st["rms"] = torch.zeros_like(p.data, dtype=torch.float32, device=p.data.device)
            if "mom" not in st:
                st["mom"] = torch.zeros_like(p.data, dtype=torch.float32, device=p.data.device)
            rms = st["rms"]
            mom = st["mom"]

            rms.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            upd = g / torch.sqrt(rms + self.eps)
            if self.use_momentum:
                mom.mul_(self.beta1).add_(upd)
                upd = upd + self.beta1 * mom if self.nesterov else mom
            if self.wd > 0 and self.wd_pattern.search(name):
                upd = upd + self.wd * p.data.detach().float()
            p.data.add_(upd.to(p.data.dtype), alpha=-lr)
        return lr

    def state_dict(self):
        return {"step_count": self.step_count, "state": self.state}

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict.get("step_count", 0))
        self.state = state_dict.get("state", {})


class Agent(nn.Module):
    """DreamerV3-style agent adapted for Isaac Lab tasks."""

    banner = [
        r"---  ___                           __   ______ ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
    ]

    def __init__(self, obs_space: dict, act_space: dict, config: dict):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self._policy_obs_key = "policy" if "policy" in obs_space else ("obs" if "obs" in obs_space else None)
        self._nextprop_key = "nextprop_target" if "nextprop_target" in obs_space else self._policy_obs_key
        self._interaction_dirs = int(config.get("interaction_dirs", 12))
        self._interaction_channels = 3

        exclude = ("is_first", "is_last", "is_terminal", "reward", "nextprop_target")
        enc_space = {k: v for k, v in obs_space.items() if k not in exclude and not k.endswith("_label")}
        dec_space = {k: v for k, v in obs_space.items() if k not in exclude and not k.endswith("_label")}

        enc_cfg = config.get("enc", {})
        dyn_cfg = config.get("dyn", {})
        dec_cfg = config.get("dec", {})

        self.enc = rssm.Encoder(enc_space, **enc_cfg)
        self.dyn = rssm.RSSM(act_space, **dyn_cfg)
        self.dec = rssm.Decoder(
            dec_space,
            **dec_cfg,
            deter=dyn_cfg.get("deter", 4096),
            stoch=dyn_cfg.get("stoch", 32),
            classes=dyn_cfg.get("classes", 32),
        )

        deter = dyn_cfg.get("deter", 4096)
        stoch = dyn_cfg.get("stoch", 32)
        classes = dyn_cfg.get("classes", 32)
        self._feat_dim = deter + stoch * classes

        rew_cfg = config.get("rewhead", {})
        con_cfg = config.get("conhead", {})
        rew_dist = rew_cfg.get("output", "symlog")
        con_dist = con_cfg.get("output", "binary")
        self.rew = MLPHead(SimpleSpace(np.float32, ()), dist_type=rew_dist, **rew_cfg)
        self.con = MLPHead(SimpleSpace(bool, (), 0, 2), dist_type=con_dist, **con_cfg)

        task_head_defaults = {"layers": 2, "units": 512, "act": "silu", "norm": "rms", "outscale": 1.0}
        nextprop_cfg = task_head_defaults | config.get("nextprop_head", {})
        stuck_cfg = task_head_defaults | config.get("stuck_head", {})
        progress_cfg = task_head_defaults | config.get("progress_head", {})
        mode_cfg = task_head_defaults | config.get("mode_head", {})
        interaction_cfg = task_head_defaults | config.get("interaction_head", {})

        if self._nextprop_key is not None:
            self.nextprop = MLPHead(self.obs_space[self._nextprop_key], dist_type="mse", **nextprop_cfg)
        else:
            self.nextprop = None
        self.stuck = MLPHead(SimpleSpace(bool, (), 0, 2), dist_type="binary", **stuck_cfg)
        self.progress = MLPHead(SimpleSpace(np.float32, ()), dist_type="mse", **progress_cfg)
        self.mode = MLPHead(SimpleSpace(np.int64, (3,), discrete=True), dist_type="onehot", **mode_cfg)
        self.interaction = MLPHead(
            SimpleSpace(np.float32, (self._interaction_dirs * self._interaction_channels,)),
            dist_type="mse",
            **interaction_cfg,
        )

        pol_cfg = config.get("policy", {})
        disc_dist = config.get("policy_dist_disc", "onehot")
        cont_dist = config.get("policy_dist_cont", "trunc_normal")
        pol_dists = {k: (disc_dist if getattr(v, "discrete", False) else cont_dist) for k, v in act_space.items()}
        self.pol = MLPHead(act_space, dist_type=pol_dists, **pol_cfg)

        val_cfg = config.get("value", {})
        val_dist = val_cfg.get("output", "symlog")
        self.val = MLPHead(SimpleSpace(np.float32, ()), dist_type=val_dist, **val_cfg)
        self.slowval = SlowModel(
            MLPHead(SimpleSpace(np.float32, ()), dist_type=val_dist, **val_cfg),
            source=self.val,
            **config.get("slowvalue", {}),
        )

        self.retnorm = RunningNorm(**config.get("retnorm", {}))
        self.valnorm = RunningNorm(**config.get("valnorm", {}))
        self.advnorm = RunningNorm(**config.get("advnorm", {}))

        self.optimizer = self._make_optimizer(**config.get("opt", {}))

        scales = config.get("loss_scales", {}).copy()
        rec = scales.pop("rec", 1.0)
        scales.update({k: rec for k in dec_space})
        scales.setdefault("nextprop", 1.0)
        scales.setdefault("stuck", 0.5)
        scales.setdefault("progress", 0.5)
        scales.setdefault("mode", 0.2)
        scales.setdefault("interaction", 0.2)
        self.scales = scales

        self._imag_length = config.get("imag_length", 15)
        self._imag_last = config.get("imag_last", None)
        self._contdisc = config.get("contdisc", True)
        self._horizon = config.get("horizon", 333)
        self._reward_grad = config.get("reward_grad", False)
        self._ac_grads = config.get("ac_grads", False)
        self._replay_context = config.get("replay_context", 0)
        self._imag_loss_cfg = config.get("imag_loss", {})
        self._repval_loss = config.get("repval_loss", False)
        self._repval_grad = config.get("repval_grad", False)
        self._repl_loss_cfg = config.get("repl_loss", {})
        self._use_task_context = bool(config.get("use_task_context", True))

    def feat2tensor(self, feat: dict) -> torch.Tensor:
        deter = feat["deter"]
        stoch = feat["stoch"].reshape(*feat["stoch"].shape[:-2], -1)
        return torch.cat([deter, stoch], dim=-1)

    def _task_context(self, latent: torch.Tensor, bdims: int = 2) -> torch.Tensor:
        parts = []
        stuck_prob = self.stuck(latent, bdims).prob(1)
        progress_pred = self.progress(latent, bdims).pred()
        mode_prob = self.mode(latent, bdims).pred()
        interaction_pred = self.interaction(latent, bdims).pred()
        parts.append(sg(stuck_prob).reshape(*stuck_prob.shape[:bdims], -1))
        parts.append(sg(progress_pred).reshape(*progress_pred.shape[:bdims], -1))
        parts.append(sg(mode_prob).reshape(*mode_prob.shape[:bdims], -1))
        parts.append(sg(interaction_pred).reshape(*interaction_pred.shape[:bdims], -1))
        return torch.cat(parts, dim=-1)

    def _policy_value_input(self, feat: dict, bdims: int = 2) -> torch.Tensor:
        latent = self.feat2tensor(feat)
        if not self._use_task_context:
            return latent
        return torch.cat([latent, self._task_context(latent, bdims)], dim=-1)

    def _make_mode_target(self, obs: dict) -> torch.Tensor | None:
        if "mode_label" in obs:
            return obs["mode_label"].float()
        if "local_tracking_state" not in obs or "derived_proprio" not in obs:
            return None
        track = obs["local_tracking_state"].float()
        derived = obs["derived_proprio"].float()
        lateral = track[..., 0].abs()
        heading = track[..., 1].abs()
        stuck = derived[..., -1] > 0.5
        track_mode = (~stuck) & (lateral < 0.25) & (heading < 0.25)
        recover_mode = stuck
        rejoin_mode = ~(track_mode | recover_mode)
        return torch.stack(
            [track_mode.float(), recover_mode.float(), rejoin_mode.float()],
            dim=-1,
        )

    def _make_interaction_target(self, obs: dict) -> torch.Tensor | None:
        if "interaction_label" in obs:
            return obs["interaction_label"].float()
        if "local_reference_window" not in obs:
            return None
        window = obs["local_reference_window"].float()
        if window.shape[-1] % 3 != 0:
            return None
        pts = window.reshape(*window.shape[:-1], -1, 3)
        xy = pts[..., :2]
        dist = torch.norm(xy, dim=-1).clamp(min=1e-6)
        angle = torch.atan2(xy[..., 1], xy[..., 0])
        bins = ((angle + math.pi) / (2 * math.pi) * self._interaction_dirs).floor().long()
        bins = bins.clamp(0, self._interaction_dirs - 1)
        score = torch.exp(-dist / 2.0)
        passability = torch.zeros(*window.shape[:-1], self._interaction_dirs, device=window.device)
        for idx in range(self._interaction_dirs):
            mask = bins == idx
            masked = torch.where(mask, score, torch.zeros_like(score))
            passability[..., idx] = masked.amax(dim=-1)
        if "derived_proprio" in obs:
            stuck = obs["derived_proprio"][..., -1].float().unsqueeze(-1)
        else:
            stuck = torch.zeros_like(passability[..., :1])
        trap_risk = (1.0 - passability) * stuck
        recovery_gain = passability * stuck
        return torch.cat([passability, trap_risk, recovery_gain], dim=-1)

    def _actdict_to_tensor(self, act: dict) -> torch.Tensor:
        parts = []
        for k in self.act_space.keys():
            v = act[k]
            if not torch.is_floating_point(v):
                v = v.float()
            parts.append(v)
        return torch.cat(parts, dim=-1)

    def _tensor_to_actdict(self, act_tensor: torch.Tensor) -> dict:
        result = {}
        offset = 0
        for k, space in self.act_space.items():
            dim = int(np.prod(space.shape))
            chunk = act_tensor[..., offset : offset + dim]
            result[k] = chunk.reshape(*chunk.shape[:-1], *space.shape)
            offset += dim
        return result

    def init_carry(self, batch_size: int, device="cpu"):
        enc_carry = self.enc.initial(batch_size, device)
        dyn_carry = self.dyn.initial(batch_size, device)
        dec_carry = self.dec.initial(batch_size, device)
        prevact = {k: torch.zeros(batch_size, *v.shape, device=device) for k, v in self.act_space.items()}
        return enc_carry, dyn_carry, dec_carry, prevact

    @torch.no_grad()
    def policy(self, carry, obs: dict, mode: str = "train"):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, single=True)
        dyn_carry, dyn_entry, feat = self.dyn.observe(dyn_carry, tokens, prevact, reset, single=True)
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, _ = self.dec(dec_carry, feat, reset, single=True)
        inp = self._policy_value_input(feat, bdims=1)
        policy_out = self.pol(inp, bdims=1)
        info = {}
        # Replay-context entries (official-style) for reconstructing latent carry from replay.
        if isinstance(enc_entry, dict):
            for k, v in enc_entry.items():
                if isinstance(v, torch.Tensor):
                    info[f"ctx_enc_{k}"] = v.detach()
        if isinstance(dyn_entry, dict):
            for k, v in dyn_entry.items():
                if isinstance(v, torch.Tensor):
                    info[f"ctx_dyn_{k}"] = v.detach()
        if isinstance(dec_entry, dict):
            for k, v in dec_entry.items():
                if isinstance(v, torch.Tensor):
                    info[f"ctx_dec_{k}"] = v.detach()
        if isinstance(policy_out, dict):
            # Match official DreamerV3 policy semantics: eval also samples actions.
            chooser = (lambda v: v.sample())
            act = {k: chooser(v) for k, v in policy_out.items()}
            for k, v in policy_out.items():
                stats = v.stats()
                if "mean" in stats:
                    info[f"policy/{k}_mean"] = stats["mean"].mean().item()
                    info[f"policy/{k}_std"] = stats["std"].mean().item()
                    info[f"policy/{k}_absmean"] = stats["mean"].abs().mean().item()
                elif "prob" in stats:
                    info[f"policy/{k}_entropy_proxy"] = (-(stats["prob"] * torch.log(stats["prob"] + 1e-8))).sum(-1).mean().item()
        else:
            act = policy_out.sample()
            stats = policy_out.stats()
            if "mean" in stats:
                info["policy/action_mean"] = stats["mean"].mean().item()
                info["policy/action_std"] = stats["std"].mean().item()
                info["policy/action_absmean"] = stats["mean"].abs().mean().item()
        carry = (enc_carry, dyn_carry, dec_carry, act)
        return carry, act, info

    def train_step(self, carry, data: dict, debug: bool = False):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        enc_carry = tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, enc_carry)
        dyn_carry = tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, dyn_carry)
        dec_carry = tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, dec_carry)
        prevact = tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, prevact)

        (enc_carry, dyn_carry, dec_carry), obs, prevact_seq, stepid = self._apply_replay_context(
            (enc_carry, dyn_carry, dec_carry), prevact, data
        )

        self.optimizer.zero_grad()
        loss, metrics, new_carry, priority, replay_updates, _ = self._compute_loss(
            (enc_carry, dyn_carry, dec_carry), obs, prevact_seq, debug=debug, training=True
        )
        loss.backward()
        lr = self.optimizer.step()
        self.slowval.update()
        new_carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
        metrics["priority"] = priority.detach()
        metrics["opt/lr"] = float(lr)
        if stepid is not None:
            replay_updates["stepid"] = stepid.detach()
        return new_carry, metrics, replay_updates

    def _apply_replay_context(self, carry, prevact, data: dict):
        enc_carry, dyn_carry, dec_carry = carry
        obs = {k: data[k] for k in self.obs_space}
        stepid = data["stepid"] if "stepid" in data else None
        prepend = lambda head, seq: torch.cat([head.unsqueeze(1), seq[:, :-1]], dim=1)
        prevact_seq = {k: prepend(prevact[k], data[k]) for k in self.act_space}

        K = int(self._replay_context)
        T = obs["is_first"].shape[1]
        if K <= 0 or T <= K:
            return (enc_carry, dyn_carry, dec_carry), obs, prevact_seq, stepid

        # Normal path: drop first K context steps from obs and prepend-shifted actions.
        normal_obs = {k: v[:, K:] for k, v in obs.items()}
        normal_prevact = {k: v[:, K:] for k, v in prevact_seq.items()}
        normal_stepid = stepid[:, K:] if isinstance(stepid, torch.Tensor) else stepid

        # Replay-context path: reconstruct carry from saved entries when available.
        rep_enc_carry, rep_dyn_carry, rep_dec_carry = enc_carry, dyn_carry, dec_carry
        has_ctx_dyn = ("ctx_dyn_deter" in data) and ("ctx_dyn_stoch" in data)
        if has_ctx_dyn:
            dyn_entries = {
                "deter": data["ctx_dyn_deter"][:, :K],
                "stoch": data["ctx_dyn_stoch"][:, :K],
            }
            rep_dyn_carry = self.dyn.truncate(dyn_entries)
        else:
            # Fallback to burn-in when context entries are unavailable.
            burn_obs = {k: v[:, :K] for k, v in obs.items()}
            burn_prevact = {k: v[:, :K] for k, v in prevact_seq.items()}
            burn_reset = burn_obs["is_first"]
            with torch.no_grad():
                rep_enc_carry, _, burn_tokens = self.enc(enc_carry, burn_obs, burn_reset)
                rep_dyn_carry, _, _ = self.dyn.observe(dyn_carry, burn_tokens, burn_prevact, burn_reset)

        rep_obs = {k: v[:, K:] for k, v in obs.items()}
        # Official replay-context action alignment: action at K-1 leads into obs at K.
        rep_prevact = {k: data[k][:, K - 1 : -1] for k in self.act_space}
        rep_stepid = stepid[:, K:] if isinstance(stepid, torch.Tensor) else stepid

        first_chunk = (
            data["consec"][:, 0] == 0
            if "consec" in data and isinstance(data["consec"], torch.Tensor)
            else torch.ones(obs["is_first"].shape[0], dtype=torch.bool, device=obs["is_first"].device)
        )
        def _select(normal, replay):
            if isinstance(normal, torch.Tensor):
                mask = first_chunk.reshape(*first_chunk.shape, *([1] * (normal.dim() - 1)))
                return torch.where(mask, replay, normal)
            return normal

        enc_carry = tree_map(_select, enc_carry, rep_enc_carry)
        dyn_carry = tree_map(_select, dyn_carry, rep_dyn_carry)
        dec_carry = tree_map(_select, dec_carry, rep_dec_carry)
        obs = tree_map(_select, normal_obs, rep_obs)
        prevact_seq = tree_map(_select, normal_prevact, rep_prevact)
        if isinstance(normal_stepid, torch.Tensor) and isinstance(rep_stepid, torch.Tensor):
            stepid = _select(normal_stepid, rep_stepid)
        else:
            stepid = normal_stepid
        return (enc_carry, dyn_carry, dec_carry), obs, prevact_seq, stepid

    def _compute_loss(self, carry, obs, prevact, debug: bool = False, training: bool = True):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        B, T = reset.shape[:2]
        losses = {}
        metrics = {}


        # 读取原始观测字段，编码成时序token表征
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset)
        # dyn_carry 计算，repfeat就是RSSM中loss的feat
        dyn_carry, dyn_entries, dyn_losses, repfeat, dyn_mets = self.dyn.loss(dyn_carry, tokens, prevact, reset)
        losses.update(dyn_losses)
        metrics.update(dyn_mets)

        dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset)

        # 将deter和stoch的Dict状态拼接成tensor，并且选择是否detach
        # reward head 只是训练自己还是连同世界模型一起训练
        latent = self.feat2tensor(repfeat)
        actor_value_inp = self._policy_value_input(repfeat, bdims=2)
        inp = sg(latent, skip=self._reward_grad)
        # rewaed head loss 计算
        rew_out = self.rew(inp, 2)
        losses["rew"] = rew_out.loss(obs["reward"])

        # continuation loss
        con = (~obs["is_terminal"]).float()
        if self._contdisc:
            con = con * (1 - 1 / self._horizon)
        con_out = self.con(self.feat2tensor(repfeat), 2)
        losses["con"] = con_out.loss(con)

        # # nexrprop head loss
        # if self.nextprop is not None and self._nextprop_key in obs and T > 1:
        #     nextprop_out = self.nextprop(latent[:, :-1], 2)
        #     losses["nextprop"] = nextprop_out.loss(obs[self._nextprop_key][:, 1:])
        #     metrics["nextprop_mae"] = (nextprop_out.pred().detach() - obs[self._nextprop_key][:, 1:].detach()).abs().mean().item()

        # # stuck head 和 progress head 的损失计算，stuck head 预测是否卡住，progress head 预测前进的距离
        # if "derived_proprio" in obs:
        #     stuck_target = obs["stuck_label"] if "stuck_label" in obs else obs["derived_proprio"][..., -1:]
        #     progress_target = obs["derived_proprio"][..., 1:2]
        #     stuck_out = self.stuck(latent, 2)
        #     progress_out = self.progress(latent, 2)
        #     losses["stuck"] = stuck_out.loss(stuck_target)
        #     losses["progress"] = progress_out.loss(progress_target)
        #     metrics["stuck_prob"] = stuck_out.prob(1).mean().item()
        #     metrics["progress_pred"] = progress_out.pred().mean().item()

        # # mode head 的损失计算，mode head 预测当前的模式（正常行走、卡住、恢复中）
        # mode_target = self._make_mode_target(obs)
        # if mode_target is not None:
        #     mode_out = self.mode(latent, 2)
        #     losses["mode"] = mode_out.loss(mode_target)
        #     metrics["mode_track_prob"] = mode_out.pred()[..., 0].mean().item()
        #     metrics["mode_recover_prob"] = mode_out.pred()[..., 1].mean().item()

        # # interaction head 的损失计算，interaction head 预测周围环境的可行性、陷阱风险和恢复增益
        # interaction_target = self._make_interaction_target(obs)
        # if interaction_target is not None:
        #     interaction_out = self.interaction(latent, 2)
        #     losses["interaction"] = interaction_out.loss(interaction_target)
        #     metrics["interaction_passability"] = interaction_out.pred()[..., : self._interaction_dirs].mean().item()

        # 计算每个重建损失
        for key, recon in recons.items():
            space, value = self.obs_space[key], obs[key]
            target = value.float() / 255 if isimage(space) else value.float()
            if isinstance(recon, DistOutput):
                losses[key] = recon.loss(sg(target))
            else:
                r = recon.flatten(2)
                t = sg(target).flatten(2)
                losses[key] = 0.5 * (r - t).square().sum(-1)

        K = min(self._imag_last or T, T)
        H = self._imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)

        imag_actions = []

        # imagine函数中会调用policyfn多次，生成imag_actions列表，列表长度为imag_length+1，
        # 最后一个元素是imag_actions中前imag_length个元素的最后一个动作对应的策略输出
        def policyfn(feat):
            out = self.pol(self._policy_value_input(feat, bdims=1), 1)
            if isinstance(out, dict):
                act = {k: v.sample() for k, v in out.items()}
            else:
                act = out.sample()
            imag_actions.append(act if isinstance(act, dict) else {"action": act})
            return act

        _, imgfeat, _ = self.dyn.imagine(starts, policyfn, H)
        imgact_dict = {k: torch.stack([a[k] for a in imag_actions], dim=1) for k in imag_actions[0].keys()}

        # 拼接出完整的imagined序列，feat状态表示
        first = tree_map(lambda x: x[:, -K:].reshape(B * K, 1, *x.shape[2:]), repfeat)
        imgfeat_full = {}
        for k in imgfeat:
            imgfeat_full[k] = torch.cat([sg(first[k], skip=self._ac_grads), sg(imgfeat[k])], dim=1)

        last_feat = {k: v[:, -1] for k, v in imgfeat_full.items()}
        lastact_dict = policyfn(last_feat)
        imag_actions.pop()

        # 完整的imagined序列，动作表示
        imgact_full = {}
        for k in imgact_dict:
            imgact_full[k] = torch.cat([imgact_dict[k], lastact_dict[k].unsqueeze(1)], dim=1)

        latent_imag = self.feat2tensor(imgfeat_full)
        imag_actor_value_inp = self._policy_value_input(imgfeat_full, bdims=2)
        imag_rew = self.rew(latent_imag, 2).pred().squeeze(-1)
        imag_con = self.con(latent_imag, 2).prob(1).squeeze(-1)

        imag_losses, imag_outs, imag_mets = imag_loss(
            imgact_full,
            imag_rew,
            imag_con,
            self.pol(imag_actor_value_inp, 2),
            self.val(imag_actor_value_inp, 2),
            self.slowval(imag_actor_value_inp, 2),
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self._contdisc,
            horizon=self._horizon,
            **self._imag_loss_cfg,
        )

        for k, v in imag_losses.items():
            losses[k] = v.mean(1).reshape(B, K)
        metrics.update(imag_mets)

        if self._repval_loss:
            feat = sg(repfeat, skip=self._repval_grad)
            last = obs["is_last"]
            term = obs["is_terminal"]
            rew = obs["reward"].squeeze(-1) if obs["reward"].shape[-1] == 1 else obs["reward"]
            boot = imag_outs["ret"][:, 0].reshape(B, K)
            feat = tree_map(lambda x: x[:, -K:], feat)
            last = last[:, -K:]
            term = term[:, -K:]
            rew = rew[:, -K:]
            boot = boot[:, -K:]
            rep_inp = self.feat2tensor(feat)
            rep_value = self.val(rep_inp, 2)
            rep_slow = self.slowval(rep_inp, 2)
            repval_losses, _, repval_metrics = repl_loss(
                last,
                term,
                rew,
                boot,
                rep_value,
                rep_slow,
                self.valnorm,
                update=training,
                horizon=self._horizon,
                **self._repl_loss_cfg,
            )
            losses.update(repval_losses)
            metrics.update({f"repval/{k}": v for k, v in repval_metrics.items()})

        if debug:
            reward_target = obs["reward"].detach().float()
            reward_pred = rew_out.pred().detach().float()
            if reward_target.dim() > reward_pred.dim():
                reward_target = reward_target.squeeze(-1)
            if reward_pred.dim() > reward_target.dim():
                reward_pred = reward_pred.squeeze(-1)
            con_target = con.detach().float()
            con_pred = con_out.prob(1).detach().float()
            if con_target.dim() > con_pred.dim():
                con_target = con_target.squeeze(-1)
            if con_pred.dim() > con_target.dim():
                con_pred = con_pred.squeeze(-1)
            feat_tensor = self.feat2tensor(repfeat).detach().float()
            rew_param_sq = 0.0
            rew_grad_sq = 0.0
            rew_grad_count = 0
            for name, param in self.rew.named_parameters():
                rew_param_sq += param.detach().float().pow(2).sum().item()
                if param.grad is not None:
                    rew_grad_sq += param.grad.detach().float().pow(2).sum().item()
                    rew_grad_count += 1
            metrics.update(
                {
                    "debug/reward_target_mean": reward_target.mean().item(),
                    "debug/reward_target_std": reward_target.std().item(),
                    "debug/reward_pred_mean": reward_pred.mean().item(),
                    "debug/reward_pred_std": reward_pred.std().item(),
                    "debug/reward_abs_err": (reward_pred - reward_target).abs().mean().item(),
                    "debug/con_target_mean": con_target.mean().item(),
                    "debug/con_pred_mean": con_pred.mean().item(),
                    "debug/is_first_rate": obs["is_first"].float().mean().item(),
                    "debug/is_last_rate": obs["is_last"].float().mean().item(),
                    "debug/is_terminal_rate": obs["is_terminal"].float().mean().item(),
                    "debug/action_abs_mean": torch.cat(
                        [prevact[k].detach().float().reshape(-1, prevact[k].shape[-1]) for k in self.act_space], dim=-1
                    ).abs().mean().item(),
                    "debug/feat_abs_mean": feat_tensor.abs().mean().item(),
                    "debug/feat_std": feat_tensor.std().item(),
                    "debug/imag_rew_mean": imag_rew.mean().item(),
                    "debug/imag_con_mean": imag_con.mean().item(),
                    "debug/rew_head_param_norm": rew_param_sq ** 0.5,
                    "debug/rew_head_grad_norm": rew_grad_sq ** 0.5 if rew_grad_count > 0 else 0.0,
                }
            )

        metrics.update({f"loss/{k}": v.mean().item() for k, v in losses.items()})
        total_loss = sum(v.mean() * self.scales.get(k, 1.0) for k, v in losses.items())
        model_keys = [k for k in ("dyn", "rep", "rew", "con") if k in losses]
        if model_keys:
            prio = torch.zeros(B, device=reset.device)
            for k in model_keys:
                v = losses[k]
                if v.dim() >= 2:
                    prio = prio + v.detach().mean(dim=1)
                else:
                    prio = prio + v.detach()
            prio = prio / float(len(model_keys))
        else:
            prio = total_loss.detach().expand(B)

        # 树状tensor detach，防止梯度流到RSSM的carry中
        carry_out = (
            tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, enc_carry),
            tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, dyn_carry),
            tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, dec_carry),
        )
        replay_updates = {
            "ctx_dyn_deter": dyn_entries["deter"].detach(),
            "ctx_dyn_stoch": dyn_entries["stoch"].detach(),
        }
        return total_loss, metrics, carry_out, prio, replay_updates, losses

    def _make_optimizer(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r"/kernel$",
        schedule: str = "const",
        warmup: int = 1000,
        anneal: int = 0,
        **kwargs,
    ) -> DreamerOptimizer:
        return DreamerOptimizer(
            list(self.named_parameters()),
            lr=lr,
            agc=agc,
            eps=eps,
            beta1=beta1,
            beta2=beta2,
            momentum=momentum,
            nesterov=nesterov,
            wd=wd,
            wdregex=wdregex,
            schedule=schedule,
            warmup=warmup,
            anneal=anneal,
        )

    def get_action(self, carry, obs: dict) -> Tuple[Any, dict, dict]:
        return self.policy(carry, obs, mode="eval")

    def update(self, carry, batch: dict):
        return self.train_step(carry, batch)

    def report_step(self, carry, data: dict):
        if not self.config.get("report", True):
            return carry, {}

        enc_carry, dyn_carry, dec_carry, prevact = carry
        (enc_carry, dyn_carry, dec_carry), obs, prevact_seq, _ = self._apply_replay_context(
            (enc_carry, dyn_carry, dec_carry), prevact, data
        )
        B, T = obs["is_first"].shape
        metrics = {}

        # Same forward/loss path as train (without optimization) for report semantics.
        _, train_metrics, (new_enc, new_dyn, new_dec), _, _, report_losses = self._compute_loss(
            (enc_carry, dyn_carry, dec_carry), obs, prevact_seq, debug=False, training=False
        )
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                metrics[f"report/{k}"] = float(v)

        if self.config.get("report_gradnorms", False):
            for key, val in report_losses.items():
                try:
                    target = val.mean() * self.scales.get(key, 1.0)
                    grads = torch.autograd.grad(
                        target,
                        [p for p in self.parameters() if p.requires_grad],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    sq = 0.0
                    for g in grads:
                        if g is not None:
                            sq += g.detach().float().pow(2).sum().item()
                    metrics[f"report/gradnorm/{key}"] = sq ** 0.5
                except Exception:
                    continue

        # Official-style open-loop check:
        # observe on first half, imagine on second half using replay actions.
        if T >= 2:
            with torch.no_grad():
                RB = min(6, B)
                split = T // 2
                first_obs = {k: v[:RB, :split] for k, v in obs.items()}
                second_prevact = {k: v[:RB, split:] for k, v in prevact_seq.items()}
                first_prevact = {k: v[:RB, :split] for k, v in prevact_seq.items()}
                first_reset = first_obs["is_first"]

                enc_small = tree_map(lambda x: x[:RB] if isinstance(x, torch.Tensor) else x, enc_carry)
                dyn_small = tree_map(lambda x: x[:RB] if isinstance(x, torch.Tensor) else x, dyn_carry)
                dec_carry = self.dec.initial(RB, device=first_reset.device)

                enc_small, _, first_tokens = self.enc(enc_small, first_obs, first_reset)
                dyn_small, _, obsfeat = self.dyn.observe(dyn_small, first_tokens, first_prevact, first_reset)
                _, imgfeat, _ = self.dyn.imagine(dyn_small, second_prevact, length=T - split)

                dec_carry, _, obsrecons = self.dec(dec_carry, obsfeat, first_reset)
                zero_reset = torch.zeros_like(obs["is_first"][:RB, split:])
                dec_carry, _, imgrecons = self.dec(dec_carry, imgfeat, zero_reset)

            for key in obsrecons.keys():
                pred = torch.cat([obsrecons[key], imgrecons[key]], dim=1)
                target = obs[key][:RB].float()
                if key in self.obs_space and isimage(self.obs_space[key]):
                    target = target / 255.0
                p = pred.reshape(pred.shape[0], pred.shape[1], -1)
                t = target.reshape(target.shape[0], target.shape[1], -1)
                metrics[f"report/openloop/{key}_mse"] = (p - t).square().mean().item()
                metrics[f"report/openloop/{key}_ctx_mse"] = (p[:, :split] - t[:, :split]).square().mean().item()
                metrics[f"report/openloop/{key}_pred_mse"] = (p[:, split:] - t[:, split:]).square().mean().item()

            # Official-style open-loop video grid for image keys.
            for key in getattr(self.dec, "imgkeys", []):
                true = obs[key][:RB]
                if true.dtype != torch.uint8:
                    true = torch.clamp(true * 255.0, 0, 255).to(torch.uint8)
                pred = torch.cat([obsrecons[key], imgrecons[key]], dim=1)
                pred = torch.clamp(pred * 255.0, 0, 255).to(torch.uint8)
                error = (((pred.to(torch.int32) - true.to(torch.int32) + 255) / 2).to(torch.uint8))
                video = torch.cat([true, pred, error], dim=2)  # (B, T, H*3, W, C)

                # Add colored border: green for observed half, red for imagined half.
                Bv, Tv, Hv, Wv, Cv = video.shape
                padded = torch.zeros((Bv, Tv, Hv + 4, Wv + 4, Cv), dtype=torch.uint8, device=video.device)
                padded[:, :, 2:-2, 2:-2, :] = video
                border = torch.zeros((Tv, 3), dtype=torch.uint8, device=video.device)
                border[:split] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=video.device)
                border[split:] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=video.device)
                padded[:, :, :2, :, :] = border[None, :, None, None, :]
                padded[:, :, -2:, :, :] = border[None, :, None, None, :]
                padded[:, :, :, :2, :] = border[None, :, None, None, :]
                padded[:, :, :, -2:, :] = border[None, :, None, None, :]

                tail = torch.zeros((Bv, 10, Hv + 4, Wv + 4, Cv), dtype=torch.uint8, device=video.device)
                video = torch.cat([padded, tail], dim=1)

                # (B, T, H, W, C) -> (T, H, B*W, C), same layout as official report.
                grid = video.permute(1, 2, 0, 3, 4).reshape(video.shape[1], video.shape[2], video.shape[0] * video.shape[3], video.shape[4])
                metrics[f"report/openloop/{key}"] = grid.detach().cpu().numpy()

        new_carry = (new_enc, new_dyn, new_dec, {k: data[k][:, -1] for k in self.act_space})
        return new_carry, metrics


class SimpleSpace:
    """Small substitute for elements.Space."""

    def __init__(self, dtype, shape=(), low=None, high=None, discrete=False):
        self.dtype = dtype
        self.shape = shape
        self.low = low
        self.high = high
        self.discrete = discrete


def imag_loss(
    act,
    rew,
    con,
    policy,
    value,
    slowvalue,
    retnorm,
    valnorm,
    advnorm,
    update=True,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    losses = {}
    metrics = {}

    stats_offset, stats_scale = valnorm.stats()
    # 把 vlaue 头输出还原到真实尺度
    val = value.pred().squeeze(-1) * stats_scale + stats_offset
    slowval = slowvalue.pred().squeeze(-1) * stats_scale + stats_offset
    # 决定 target 使用哪个 value
    tarval = slowval if slowtar else val

    disc = 1.0 if contdisc else 1.0 - 1.0 / horizon
    # torch.cumprod 累积乘积，沿着第一维
    weight = torch.cumprod(disc * con, dim=1) / disc
    last = torch.zeros_like(con)
    term = 1.0 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    roffset, rscale = retnorm(ret, update)
    adv_raw = ret - tarval[:, :-1]
    adv = adv_raw / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale

    if isinstance(policy, dict):
        logpi = sum(v.logp(sg(act[k]))[:, :-1] for k, v in policy.items())
        ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
    else:
        logpi = policy.logp(sg(act))[:, :-1]
        ents = {"act": policy.entropy()[:, :-1]}

    losses["policy"] = sg(weight[:, :-1]) * -(logpi * sg(adv_normed) + actent * sum(ents.values()))

    train_offset, train_scale = valnorm(ret, update)
    tar_normed = (ret - train_offset) / train_scale
    tar_padded = torch.cat([tar_normed, torch.zeros_like(tar_normed[:, -1:])], 1)
    losses["value"] = sg(weight[:, :-1]) * (
        value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred()))
    )[:, :-1]

    ret_normed = (ret - roffset) / rscale
    val_raw = val[:, :-1]
    slowval_raw = slowval[:, :-1]
    val_normed = (val_raw - train_offset) / train_scale
    slowval_normed = (slowval_raw - train_offset) / train_scale
    # Advantage diagnostics: distinguish true signal collapse from normalization squeeze.
    metrics["adv"] = adv.mean().item()
    metrics["adv_std"] = adv.std().item()
    metrics["adv_mag"] = adv.abs().mean().item()
    metrics["adv_raw"] = adv_raw.mean().item()
    metrics["adv_raw_std"] = adv_raw.std().item()
    metrics["adv_raw_mag"] = adv_raw.abs().mean().item()
    metrics["ret_slow_gap_raw"] = adv_raw.mean().item()
    metrics["ret_slow_gap_abs"] = adv_raw.abs().mean().item()
    metrics["ret_slow_gap_std"] = adv_raw.std().item()
    metrics["ret_norm_scale"] = float(rscale)
    if adv_raw.numel() > 1:
        ret_centered = ret - ret.mean()
        tar_centered = tarval[:, :-1] - tarval[:, :-1].mean()
        denom = ret_centered.std() * tar_centered.std() + 1e-8
        corr = (ret_centered * tar_centered).mean() / denom
        metrics["ret_slow_corr"] = corr.item()
    else:
        metrics["ret_slow_corr"] = 0.0
    # Note: lambda_return uses rew[:, 1:], since index 0 is the anchor state.
    metrics["rew"] = rew.mean().item()
    metrics["rew_used"] = rew[:, 1:].mean().item() if rew.shape[1] > 1 else rew.mean().item()
    metrics["con"] = con.mean().item()
    metrics["ret"] = ret_normed.mean().item()
    metrics["ret_raw"] = ret.mean().item()
    metrics["ret_value_norm"] = tar_normed.mean().item()
    metrics["val"] = val_normed.mean().item()
    metrics["val_raw"] = val_raw.mean().item()
    metrics["tar"] = tar_normed.mean().item()
    metrics["weight"] = weight.mean().item()
    # Keep naming semantically correct for debugging.
    metrics["tar_raw"] = tarval[:, :-1].mean().item()
    metrics["slowval"] = slowval_normed.mean().item()
    metrics["slowval_raw"] = slowval_raw.mean().item()
    metrics["ret_min"] = ret_normed.min().item()
    metrics["ret_max"] = ret_normed.max().item()
    metrics["ret_rate"] = (ret_normed.abs() >= 1.0).float().mean().item()
    # Consistency check: lambda-return should satisfy its own Bellman-style recursion.
    live = (1 - term[:, 1:].float()) * disc
    cont = (1 - last[:, 1:].float()) * lam
    next_val = tarval[:, 1:]
    ret_next = torch.cat([ret[:, 1:], tarval[:, -1:]], dim=1)
    ret_target = rew[:, 1:] + live * ((1 - cont) * next_val + cont * ret_next)
    ret_residual = ret - ret_target
    metrics["lambda_residual_absmean"] = ret_residual.abs().mean().item()
    metrics["lambda_residual_maxabs"] = ret_residual.abs().max().item()
    for k in ents:
        metrics[f"ent/{k}"] = ents[k].mean().item()
        if isinstance(policy, dict) and hasattr(policy.get(k), "minent"):
            lo, hi = policy[k].minent, policy[k].maxent
            metrics[f"rand/{k}"] = ((ents[k].mean() - lo) / (hi - lo)).item()

    return losses, {"ret": ret}, metrics


def repl_loss(
    last,
    term,
    rew,
    boot,
    value,
    slowvalue,
    valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
    losses = {}
    metrics = {}
    if last.dim() > 0 and last.shape[-1] == 1:
        last = last.squeeze(-1)
    if term.dim() > 0 and term.shape[-1] == 1:
        term = term.squeeze(-1)
    if rew.dim() > 0 and rew.shape[-1] == 1:
        rew = rew.squeeze(-1)
    stats_offset, stats_scale = valnorm.stats()
    val = value.pred().squeeze(-1) * stats_scale + stats_offset
    slowval = slowvalue.pred().squeeze(-1) * stats_scale + stats_offset
    tarval = slowval if slowtar else val
    disc = 1.0 - 1.0 / horizon
    weight = (~last).float()
    if isinstance(boot, DistOutput):
        boot_pred = boot.pred().squeeze(-1) * stats_scale + stats_offset
    else:
        boot_pred = boot.squeeze(-1) if boot.dim() > 0 and boot.shape[-1] == 1 else boot
    ret = lambda_return(last.float(), term.float(), rew, tarval, boot_pred, disc, lam)

    train_offset, train_scale = valnorm(ret, update)
    ret_normed = (ret - train_offset) / train_scale
    ret_padded = torch.cat([ret_normed, torch.zeros_like(ret_normed[:, -1:])], 1)
    losses["repval"] = weight[:, :-1] * (
        value.loss(sg(ret_padded)) + slowreg * value.loss(sg(slowvalue.pred()))
    )[:, :-1]
    val_raw = val[:, :-1]
    slowval_raw = slowval[:, :-1]
    val_normed = (val_raw - train_offset) / train_scale
    slowval_normed = (slowval_raw - train_offset) / train_scale
    metrics["ret"] = ret_normed.mean().item()
    metrics["ret_raw"] = ret.mean().item()
    metrics["val"] = val_normed.mean().item()
    metrics["val_raw"] = val_raw.mean().item()
    metrics["tar"] = ret_normed.mean().item()
    metrics["tar_raw"] = ret.mean().item()
    metrics["slowval"] = slowval_normed.mean().item()
    metrics["slowval_raw"] = slowval_raw.mean().item()
    # Consistency check for replay lambda-return recursion.
    live = (1 - term[:, 1:].float()) * disc
    cont = (1 - last[:, 1:].float()) * lam
    next_boot = boot_pred[:, 1:]
    ret_next = torch.cat([ret[:, 1:], boot_pred[:, -1:]], dim=1)
    ret_target = rew[:, 1:] + live * ((1 - cont) * next_boot + cont * ret_next)
    ret_residual = ret - ret_target
    metrics["lambda_residual_absmean"] = ret_residual.abs().mean().item()
    metrics["lambda_residual_maxabs"] = ret_residual.abs().max().item()
    return losses, {"ret": ret}, metrics



"""
    Function: 把即时 reward、bootstrap value、折扣因子和 λ 混合起来，递归构造一个多步但不过分高方差的回报目标。
    disc: 折扣因子 gamma
    boot: bootstrap value
    lam: lambda
"""
def lambda_return(last, term, rew, val, boot, disc, lam):
    if last.dim() > 0 and last.shape[-1] == 1:
        last = last.squeeze(-1)
    if term.dim() > 0 and term.shape[-1] == 1:
        term = term.squeeze(-1)
    if rew.dim() > 0 and rew.shape[-1] == 1:
        rew = rew.squeeze(-1)
    if val.dim() > 0 and val.shape[-1] == 1:
        val = val.squeeze(-1)
    if boot.dim() > 0 and boot.shape[-1] == 1:
        boot = boot.squeeze(-1)
    rets = [boot[:, -1]]
    live = (1 - term[:, 1:].float()) * disc
    cont = (1 - last[:, 1:].float()) * lam
    next_boot = boot[:, 1:]
    interm = rew[:, 1:] + (1 - cont) * live * next_boot
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return torch.stack(list(reversed(rets))[:-1], dim=1)
