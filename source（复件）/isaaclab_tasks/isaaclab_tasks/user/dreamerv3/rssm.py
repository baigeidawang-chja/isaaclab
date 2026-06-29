import math
from typing import Dict, Optional, Tuple, Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from einops import rearrange


# ─────────────────────────────────────────────
# 辅助函数 / 模块
# ─────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def get_act(name: str):
    return {
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'elu': nn.ELU,
        'tanh': nn.Tanh,
        'none': nn.Identity,
    }[name]


def get_norm(name: str, dim: int):
    if name == 'rms':
        # 均方根层归一化
        return nn.RMSNorm(dim)
    elif name == 'layer':
        return nn.LayerNorm(dim)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {name}")


class DictConcat(nn.Module):
    """将字典中的多个张量沿最后一维拼接，可选 squish 变换。"""
    def __init__(self, space_dims: Dict[str, int], squish=None):
        """
        Args:
            space_dims: {key: feature_dim} 映射
            squish: 对连续特征施加的变换(如 symlog)
        """
        super().__init__()
        self.space_dims = space_dims
        self.squish = squish

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for k in sorted(self.space_dims.keys()):
            x = inputs[k]
            if self.squish is not None:
                x = self.squish(x)
            if x.dim() > 1 and x.shape[-1] != self.space_dims[k]:
                x = x.reshape(*x.shape[:-1], -1)
            parts.append(x.float())
        return torch.cat(parts, dim=-1)


class LinearLayer(nn.Module):
    """
    带可选 outscale 初始化的线性层。
    in_dim: 输入维度
    out_dim: 输出维度
    outscale: 输出缩放因子，控制初始化范围
    """
    
    def __init__(self, in_dim: int, out_dim: int, outscale: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # fan-in 初始化，按 outscale 缩放
        with torch.no_grad():
            fan_in = self.linear.weight.shape[1]
            std = outscale / math.sqrt(fan_in)
            self.linear.weight.normal_(0, std)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BlockLinear(nn.Module):
    """
    分块线性层：
    输入分成 g 组，每组独立做线性变换。
    然后再reshape回来
    """
    def __init__(self, in_dim: int, out_dim: int, groups: int, outscale: float = 1.0):
        super().__init__()
        assert in_dim % groups == 0 and out_dim % groups == 0
        self.groups = groups
        self.in_per_group = in_dim // groups
        self.out_per_group = out_dim // groups
        self.weight = nn.Parameter(torch.empty(groups, self.in_per_group, self.out_per_group))
        self.bias = nn.Parameter(torch.zeros(groups, self.out_per_group))
        # 初始化
        with torch.no_grad():
            std = outscale / math.sqrt(self.in_per_group)
            self.weight.normal_(0, std)
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        shape = x.shape[:-1]
        g = self.groups
        x = x.reshape(*shape, g, self.in_per_group)      # (..., g, in_g)
        x = torch.einsum('...gi,gio->...go', x, self.weight) + self.bias
        return x.reshape(*shape, g * self.out_per_group)  # (..., out_dim)


class Conv2DLayer(nn.Module):
    """封装的 2D 卷积 / 转置卷积。"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 stride: int = 1, transp: bool = False, outscale: float = 1.0):
        super().__init__()
        padding = kernel // 2
        if transp:
            self.conv = nn.ConvTranspose2d(
                in_ch, out_ch, kernel, stride=stride,
                padding=padding, output_padding=stride - 1)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        with torch.no_grad():
            fan_in = self.conv.weight.shape[1] * self.conv.weight.shape[2] * self.conv.weight.shape[3]
            std = outscale / math.sqrt(fan_in)
            self.conv.weight.normal_(0, std)
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, layers: int, units: int,
                 norm: str = 'rms', act: str = 'gelu', outscale: float = 1.0):
        super().__init__()
        mods = []
        d = in_dim
        for i in range(layers):
            mods.append(LinearLayer(d, units, outscale=1.0))
            mods.append(get_norm(norm, units))
            mods.append(get_act(act)())
            d = units
        self.net = nn.Sequential(*mods)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# OneHot 分布 (直通梯度 + unimix)
# ─────────────────────────────────────────────

class OneHotStraightThrough:
    """OneHot 采样 + 直通梯度，附带 unimix。"""
    def __init__(self, logits: torch.Tensor, unimix: float = 0.01):
        self.classes = logits.shape[-1]
        if unimix > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.classes
            probs = (1 - unimix) * probs + unimix * uniform
            self.logits = torch.log(probs + 1e-8)
        else:
            self.logits = logits
        self._dist = td.OneHotCategorical(logits=self.logits)

    def sample(self) -> torch.Tensor:
        sample = self._dist.sample()
        # 直通梯度
        probs = F.softmax(self.logits, dim=-1)
        return sample + probs - probs.detach()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def kl(self, other: 'OneHotStraightThrough') -> torch.Tensor:
        return td.kl_divergence(
            td.OneHotCategorical(logits=self.logits),
            td.OneHotCategorical(logits=other.logits))


class AggDist:
    """对多个独立分布在指定维度上聚合 (求和) KL / entropy。"""
    def __init__(self, base_dists: list, agg_dim: int):
        self.base_dists = base_dists
        self.agg_dim = agg_dim

    @staticmethod
    def from_logits(logits: torch.Tensor, unimix: float = 0.01):
        """logits: (..., stoch, classes)"""
        # 对 stoch 维度逐个构建 OneHotStraightThrough
        return _AggOneHot(logits, unimix)


class _AggOneHot:
    """
    处理 (..., stoch, classes) 形状 logits 的聚合分布。
    对 stoch 维度求和 KL / entropy。
    """
    def __init__(self, logits: torch.Tensor, unimix: float = 0.01):
        self.logits = logits
        self.unimix = unimix
        self._dist = OneHotStraightThrough(logits, unimix)

    def sample(self) -> torch.Tensor:
        return self._dist.sample()

    def entropy(self) -> torch.Tensor:
        # 每个 stoch slot 的 entropy, 对 stoch 维求和
        return self._dist.entropy().sum(dim=-1)

    def kl(self, other: '_AggOneHot') -> torch.Tensor:
        return self._dist.kl(other._dist).sum(dim=-1)


# ─────────────────────────────────────────────
# RSSM 核心
# ─────────────────────────────────────────────

class RSSM(nn.Module):
    def __init__(
        self,
        act_space: Dict[str, 'Space'],
        deter: int = 4096,
        hidden: int = 2048,
        stoch: int = 32,
        classes: int = 32,
        norm: str = 'rms',
        act: str = 'gelu',
        unimix: float = 0.01,
        outscale: float = 1.0,
        imglayers: int = 2,
        obslayers: int = 1,
        dynlayers: int = 1,
        absolute: bool = False,
        blocks: int = 8,
        free_nats: float = 1.0,
    ):
        super().__init__()
        assert deter % blocks == 0
        self.act_space = act_space
        self._deter = deter
        self._hidden = hidden
        self._stoch = stoch
        self._classes = classes
        self._norm = norm
        self._act = act
        self._unimix = unimix
        self._outscale = outscale
        self._imglayers = imglayers
        self._obslayers = obslayers
        self._dynlayers = dynlayers
        self._absolute = absolute
        self._blocks = blocks
        self._free_nats = free_nats

        # 计算 act_space 拼接后的维度
        self._act_dim = sum(
            s.shape[0] if hasattr(s, 'shape') and len(s.shape) > 0 else 1
            for s in act_space.values()
        )

        # ── 动态核心网络 ──
        self.dynin0 = LinearLayer(deter, hidden)
        self.dynin0norm = get_norm(norm, hidden)
        self.dynin1 = LinearLayer(stoch * classes, hidden)
        self.dynin1norm = get_norm(norm, hidden)
        self.dynin2 = LinearLayer(self._act_dim, hidden)
        self.dynin2norm = get_norm(norm, hidden)

        core_in = deter + blocks * (3 * hidden)   # deter(分组) + [x0,x1,x2] repeat
        self.dynhids = nn.ModuleList()
        self.dynhid_norms = nn.ModuleList()
        d = core_in
        for i in range(dynlayers):
            self.dynhids.append(BlockLinear(d, deter, blocks))
            self.dynhid_norms.append(get_norm(norm, deter))
            d = deter
        self.dyngru = BlockLinear(deter, 3 * deter, blocks)

        # ── 观测后验网络 ──
        # obs 输入维度在首次前向时确定
        self._obs_in_dim = None
        self.obs_layers = nn.ModuleList()
        self.obs_norms = nn.ModuleList()
        # 占位：将在 _lazy_init_obs 中构建
        self._obs_built = False

        # ── 先验网络 ──
        self.prior_layers = nn.ModuleList()
        self.prior_norms = nn.ModuleList()
        d = deter
        for i in range(imglayers):
            self.prior_layers.append(LinearLayer(d, hidden))
            self.prior_norms.append(get_norm(norm, hidden))
            d = hidden
        self.prior_logit = LinearLayer(d, stoch * classes, outscale=outscale)

        # 后验 logit 层 (同样延迟构建)
        self._obs_logit = None
        self._act_fn = get_act(act)

    def _lazy_init_obs(self, obs_in_dim: int):
        """延迟构建观测后验网络（因为 token 维度取决于编码器）。"""
        d = obs_in_dim
        for i in range(self._obslayers):
            self.obs_layers.append(LinearLayer(d, self._hidden).to(self._device))
            self.obs_norms.append(get_norm(self._norm, self._hidden).to(self._device))
            d = self._hidden
        self._obs_logit = LinearLayer(d, self._stoch * self._classes,
                                       outscale=self._outscale).to(self._device)
        self._obs_built = True

    @property
    def _device(self):
        return self.dynin0.linear.weight.device

    # ── 初始化 carry ──
    def initial(self, batch_size: int, device='cpu') -> Dict[str, torch.Tensor]:
        return dict(
            deter=torch.zeros(batch_size, self._deter, device=device),
            stoch=torch.zeros(batch_size, self._stoch, self._classes, device=device),
        )

    # ── 截断 ──
    def truncate(self, entries: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # entries: (B, T, ...)  →  取最后一步作为 carry
        return {k: v[:, -1] for k, v in entries.items()}


    """
        从一整段真实序列的latent状态里，取出最后nlast个时间步，摊平成一批imagined rollout的起点
        Return: {
                    "deter": (B*nlast, D),
                    "stoch": (B*nlast, S, C),
                }
    """
    def starts(self, entries: Dict[str, torch.Tensor],
               carry: Dict[str, torch.Tensor], nlast: int):
        B = carry['deter'].shape[0]
        return {
            k: v[:, -nlast:].reshape(B * nlast, *v.shape[2:])
            for k, v in entries.items()
        }

    # ── 掩码 ──
    @staticmethod
    def _mask(tensors, mask):
        """mask: (B,) bool, True 表示保留。"""
        if mask is not None:
            return tuple(
                t * mask.reshape(*mask.shape, *([1] * (t.dim() - mask.dim())))
                for t in tensors
            )
        return tensors

    # ── 拼接 action ──
    def _concat_action(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [action[k].float() for k in sorted(action.keys())]
        return torch.cat(parts, dim=-1)

    # ── GRU 动态核心 ──
    def _core(self, deter: torch.Tensor, stoch: torch.Tensor,
              action: torch.Tensor) -> torch.Tensor:
        """
        deter: (B, deter)
        stoch: (B, stoch*classes)
        action: (B, act_dim)
        Returns: (B, deter)
        """

        # 分组数
        g = self._blocks

        # 激活函数，例如GELU
        act_fn = self._act_fn()

        # 动态输入处理：分别处理 deter,stoch,action
        # 每一个都是(B, hidden)
        x0 = act_fn(self.dynin0norm(self.dynin0(deter)))
        x1 = act_fn(self.dynin1norm(self.dynin1(stoch)))
        x2 = act_fn(self.dynin2norm(self.dynin2(action)))

        # 拼接后 repeat 到每个 block 组
        combined = torch.cat([x0, x1, x2], dim=-1)  # (B, 3*hidden)
        # (B, g, 3*hidden)
        combined = combined.unsqueeze(-2).expand(*combined.shape[:-1], g, combined.shape[-1])
        # deter 分组
        deter_grouped = rearrange(deter, '... (g h) -> ... g h', g=g)
        x = torch.cat([deter_grouped, combined], dim=-1)  # (B, g, deter/g + 3*hidden)
        x = rearrange(x, '... g h -> ... (g h)', g=g)

        # 这里进行了
        # 一次分组线性化
        for layer, norm in zip(self.dynhids, self.dynhid_norms):
            x = act_fn(norm(layer(x)))

        x = self.dyngru(x)  # (B, 3*deter)
        # 将输出分为三部分：重置门、候选状态、更新门
        # (B, g, deter/g)
        gates = rearrange(x, '... (g h) -> ... g h', g=g).chunk(3, dim=-1)
        reset_gate, cand, update = [rearrange(t, '... g h -> ... (g h)', g=g) for t in gates]


        # 重置门： 控制历史信息保留
        reset_gate = torch.sigmoid(reset_gate)
        # 候选状态： 新信息
        cand = torch.tanh(reset_gate * cand)
        # 更新门： 控制新旧信息混合，减去1使初始值接近0
        update = torch.sigmoid(update - 1)
        # GRU 更新公式
        deter = update * cand + (1 - update) * deter
        return deter

    # ── logit 映射 ──
    def _logit(self, linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = linear(x)
        return x.reshape(*x.shape[:-1], self._stoch, self._classes)

    # ── 分布 ──
    def _dist(self, logits: torch.Tensor) -> _AggOneHot:
        return _AggOneHot(logits, self._unimix)

    # ── 先验 ──
    def _prior(self, deter: torch.Tensor) -> torch.Tensor:
        act_fn = self._act_fn()
        x = deter
        for layer, norm in zip(self.prior_layers, self.prior_norms):
            x = act_fn(norm(layer(x)))
        return self._logit(self.prior_logit, x)

    # ── 单步 observe ──
    def _observe_step(self, carry: Dict[str, torch.Tensor],
                      tokens: torch.Tensor,
                      action: torch.Tensor,
                      reset: torch.Tensor):
        """
        carry: {deter: (B, D), stoch: (B, S, C)}
        tokens: (B, token_dim)
        action: (B, act_dim)  已拼接
        reset: (B,) bool

        Function: RSSM在单个时间步上，用真实观测更新laten state的核心。
        """
        mask = ~reset
        deter, stoch, action = self._mask(
            (carry['deter'], carry['stoch'], action), mask)

        # 把 stoch 沿着批量维展平
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        # symlog-normalize action
        action = action / torch.clamp(torch.abs(action), min=1.0)
        # 根据上一时刻的latent和动作，更新当前的deteministic hidden state
        deter = self._core(deter, stoch_flat, action)
        # deter展平
        tokens_flat = tokens.reshape(*deter.shape[:-1], -1)

        if self._absolute:
            x = tokens_flat
            obs_in_dim = tokens_flat.shape[-1]
        else:
            x = torch.cat([deter, tokens_flat], dim=-1)
            obs_in_dim = self._deter + tokens_flat.shape[-1]

        # 延迟构建 obs 层
        if not self._obs_built:
            self._lazy_init_obs(obs_in_dim)

        # posterior MLP
        act_fn = self._act_fn()
        for layer, norm in zip(self.obs_layers, self.obs_norms):
            x = act_fn(norm(layer(x)))

        # 线性层处理然后reshape
        logit = self._logit(self._obs_logit, x)
        # 有logit定义出来的概率分布对象
        dist = self._dist(logit)
        # 从分布中采样
        stoch = dist.sample()

        # 需要滚动的， 即下一时间步递归更新所必需的状态
        new_carry = dict(deter=deter, stoch=stoch)
        # 给后续loss/head/统计使用的更完整特征包
        feat = dict(deter=deter, stoch=stoch, logit=logit)
        # 这一时刻需要被记录下来的latent state快照
        entry = dict(deter=deter, stoch=stoch)
        return new_carry, entry, feat

    # ── observe (序列) ──
    def observe(self, carry: Dict[str, torch.Tensor],
                tokens: torch.Tensor,
                action: Dict[str, torch.Tensor],
                reset: torch.Tensor,
                single: bool = False):
        """
        carry: {deter: (B, D), stoch: (B, S, C)}
        tokens: (B, [T,] token_dim)
        action: dict of (B, [T,] ...)
        reset: (B, [T])
        single: 如果 True，则没有 T 维度
        """
        action_cat = self._concat_action(action)
        # normalize action
        action_cat = action_cat / torch.clamp(torch.abs(action_cat), min=1.0)

        if single:
            new_carry, entry, feat = self._observe_step(
                carry, tokens, action_cat, reset)
            return new_carry, entry, feat
        else:
            B, T = reset.shape[:2]
            entries = {k: [] for k in ('deter', 'stoch')}
            feats = {k: [] for k in ('deter', 'stoch', 'logit')}

            for t in range(T):
                tok_t = tokens[:, t]
                act_t = action_cat[:, t]
                rst_t = reset[:, t]
                carry, entry, feat = self._observe_step(carry, tok_t, act_t, rst_t)
                for k in entries:
                    entries[k].append(entry[k])
                for k in feats:
                    feats[k].append(feat[k])

            entries = {k: torch.stack(v, dim=1) for k, v in entries.items()}
            feats = {k: torch.stack(v, dim=1) for k, v in feats.items()}
            return carry, entries, feats

    # ── 单步 imagine ──
    def _imagine_step(self, carry: Dict[str, torch.Tensor],
                      action: torch.Tensor):
        """
        action: (B, act_dim) 已拼接
        """
        stoch_flat = carry['stoch'].reshape(carry['stoch'].shape[0], -1)
        action = action / torch.clamp(torch.abs(action), min=1.0)
        deter = self._core(carry['deter'], stoch_flat, action)
        logit = self._prior(deter)
        dist = self._dist(logit)
        stoch = dist.sample()
        new_carry = dict(deter=deter, stoch=stoch)
        feat = dict(deter=deter, stoch=stoch, logit=logit)
        return new_carry, feat

    # ── imagine (序列) ──
    def imagine(self, carry: Dict[str, torch.Tensor],
                policy: Union[Callable, Dict[str, torch.Tensor]],
                length: int,
                single: bool = False):
        """
        policy: 可调用 (接收 carry, 返回 action dict)，或预计算的 action dict
        """
        if single:
            if callable(policy):
                with torch.no_grad():
                    action = policy({k: v.detach() for k, v in carry.items()})
            else:
                action = policy
            action_cat = self._concat_action(action)
            new_carry, feat = self._imagine_step(carry, action_cat)
            return new_carry, feat, action
        else:
            feats = {k: [] for k in ('deter', 'stoch', 'logit')}
            actions = []
            for t in range(length):
                if callable(policy):
                    with torch.no_grad():
                        action = policy({k: v.detach() for k, v in carry.items()})
                    action_cat = self._concat_action(action)
                else:
                    action_cat = self._concat_action(
                        {k: v[:, t] for k, v in policy.items()})
                    action = {k: v[:, t] for k, v in policy.items()}
                carry, feat = self._imagine_step(carry, action_cat)
                for k in feats:
                    feats[k].append(feat[k])
                actions.append(action_cat)
            feats = {k: torch.stack(v, dim=1) for k, v in feats.items()}
            actions = torch.stack(actions, dim=1)
            return carry, feats, actions

    # ── loss ──
    def loss(self, carry: Dict[str, torch.Tensor],
             tokens: torch.Tensor,
             acts: Dict[str, torch.Tensor],
             reset: torch.Tensor):
        """
        Returns: carry, entries, losses, feat, metrics
        """
        metrics = {}
        carry, entries, feat = self.observe(carry, tokens, acts, reset)
        # 先验
        prior = self._prior(feat['deter'])
        # 后验
        post = feat['logit']

        # 计算两个方向不同、梯度流向不同的项
        # 训练prior/dynamics网络
        dyn = self._dist(post.detach()).kl(self._dist(prior))
        # 训练posterior/representation网络
        rep = self._dist(post).kl(self._dist(prior.detach()))

        if self._free_nats:
            dyn = torch.clamp(dyn, min=self._free_nats)
            rep = torch.clamp(rep, min=self._free_nats)

        losses = {'dyn': dyn, 'rep': rep}
        metrics['dyn_ent'] = self._dist(prior).entropy().mean()
        metrics['rep_ent'] = self._dist(post).entropy().mean()
        return carry, entries, losses, feat, metrics


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(
        self,
        obs_space: Dict[str, 'Space'],
        units: int = 1024,
        norm: str = 'rms',
        act: str = 'gelu',
        depth: int = 64,
        mults: Tuple[int, ...] = (2, 3, 4, 4),
        layers: int = 3,
        kernel: int = 5,
        symlog_input: bool = True,
        outer: bool = False,
        strided: bool = False,
    ):
        super().__init__()
        self.obs_space = obs_space
        self._units = units
        self._norm = norm
        self._act = act
        self._depth = depth
        self._mults = mults
        self._layers = layers
        self._kernel = kernel
        self._symlog = symlog_input
        self._outer = outer
        self._strided = strided

        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.depths = tuple(depth * m for m in mults)

        # ── MLP 编码器（向量特征）──
        if self.veckeys:
            vec_dim = sum(
                int(np.prod(obs_space[k].shape)) for k in self.veckeys
            )
            self.mlp_layers = nn.ModuleList()
            self.mlp_norms = nn.ModuleList()
            d = vec_dim
            for i in range(layers):
                self.mlp_layers.append(LinearLayer(d, units))
                self.mlp_norms.append(get_norm(norm, units))
                d = units
            self._vec_out_dim = d
        else:
            self._vec_out_dim = 0

        # ── CNN 编码器（图像特征）──
        if self.imgkeys:
            img_ch = sum(obs_space[k].shape[-1] for k in sorted(self.imgkeys))
            self.cnn_layers = nn.ModuleList()
            self.cnn_norms = nn.ModuleList()
            in_ch = img_ch
            for i, dep in enumerate(self.depths):
                if outer and i == 0:
                    self.cnn_layers.append(
                        Conv2DLayer(in_ch, dep, kernel, stride=1))
                elif strided:
                    self.cnn_layers.append(
                        Conv2DLayer(in_ch, dep, kernel, stride=2))
                else:
                    self.cnn_layers.append(
                        Conv2DLayer(in_ch, dep, kernel, stride=1))
                self.cnn_norms.append(get_norm(norm, dep))
                in_ch = dep
            # 输出维度延迟计算
            self._cnn_out_dim = None
        else:
            self._cnn_out_dim = 0

    @property
    def out_dim(self):
        """编码器输出 token 维度（需在首次前向后确定 CNN 部分）。"""
        if self._cnn_out_dim is None:
            return None
        return self._vec_out_dim + self._cnn_out_dim

    def initial(self, batch_size: int, device='cpu'):
        return {}

    def truncate(self, entries):
        return {}

    def forward(self, carry: dict, obs: Dict[str, torch.Tensor],
                reset: torch.Tensor, single: bool = False):
        """
        obs: dict of tensors, 图像 uint8, 向量 float
        reset: (B,) or (B, T)
        Returns: carry, entries, tokens
        """
        bdims = 1 if single else 2
        bshape = reset.shape
        outs = []
        act_fn = get_act(self._act)()

        # ── 向量 ──
        if self.veckeys:
            parts = []
            for k in sorted(self.veckeys):
                v = obs[k].float()
                if self._symlog:
                    v = symlog(v)
                parts.append(v.reshape(*v.shape[:bdims], -1))
            x = torch.cat(parts, dim=-1)
            x = x.reshape(-1, x.shape[-1])  # flatten batch dims
            for layer, norm in zip(self.mlp_layers, self.mlp_norms):
                x = act_fn(norm(layer(x)))
            outs.append(x)

        # ── 图像 ──
        if self.imgkeys:
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            x = torch.cat(imgs, dim=-1).float() / 255.0 - 0.5
            x = x.reshape(-1, *x.shape[bdims:])         # (B*, H, W, C)
            x = x.permute(0, 3, 1, 2)                    # → (B*, C, H, W)
            for i, (layer, norm) in enumerate(zip(self.cnn_layers, self.cnn_norms)):
                x = layer(x)
                if not (self._outer and i == 0) and not self._strided:
                    # max-pool 2x2
                    x = F.max_pool2d(x, 2)
                # norm 需要 (B, C) 或者按通道
                B_, C_, H_, W_ = x.shape
                x_flat = x.permute(0, 2, 3, 1).reshape(-1, C_)
                x_flat = norm(x_flat)
                x_flat = act_fn(x_flat)
                x = x_flat.reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)
            # flatten spatial
            x = x.reshape(x.shape[0], -1)
            if self._cnn_out_dim is None:
                self._cnn_out_dim = x.shape[-1]
            outs.append(x)

        x = torch.cat(outs, dim=-1)
        tokens = x.reshape(*bshape, *x.shape[1:])
        return carry, {}, tokens


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(
        self,
        obs_space: Dict[str, 'Space'],
        units: int = 1024,
        norm: str = 'rms',
        act: str = 'gelu',
        outscale: float = 1.0,
        depth: int = 64,
        mults: Tuple[int, ...] = (2, 3, 4, 4),
        layers: int = 3,
        kernel: int = 5,
        symlog_output: bool = True,
        bspace: int = 8,
        outer: bool = False,
        strided: bool = False,
        # deter: int = 4096,
        deter: int = 1024,
        stoch: int = 32,
        classes: int = 32,
    ):
        super().__init__()
        self.obs_space = obs_space
        self._units = units
        self._norm = norm
        self._act = act
        self._outscale = outscale
        self._depth = depth
        self._mults = mults
        self._layers = layers
        self._kernel = kernel
        self._symlog = symlog_output
        self._bspace = bspace
        self._outer = outer
        self._strided = strided
        self._deter = deter
        self._stoch = stoch
        self._classes = classes

        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.depths = tuple(depth * m for m in mults)
        self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
        self.imgres = (
            obs_space[self.imgkeys[0]].shape[:-1] if self.imgkeys else None
        )

        feat_dim = deter + stoch * classes

        # ── 向量解码器 ──
        if self.veckeys:
            self.vec_mlp = MLP(feat_dim, layers, units, norm, act)
            # 输出头：每个 key 一个线性层
            self.vec_heads = nn.ModuleDict()
            for k in self.veckeys:
                out_dim = int(np.prod(obs_space[k].shape))
                discrete = getattr(obs_space[k], 'discrete', False)
                self.vec_heads[k] = LinearLayer(units, out_dim, outscale=outscale)

        # ── 图像解码器 ── 
        if self.imgkeys:
            factor = 2 ** (len(self.depths) - int(bool(outer)))
            self.minres = [int(r // factor) for r in self.imgres]

            if bspace:
                g = bspace
                u = math.prod((*self.minres, self.depths[-1]))
                self.sp0 = BlockLinear(deter, u, g)
                self.sp1 = LinearLayer(stoch * classes, 2 * units)
                self.sp1norm = get_norm(norm, 2 * units)
                self.sp2 = LinearLayer(2 * units, math.prod((*self.minres, self.depths[-1])))
                self.spnorm = get_norm(norm, self.depths[-1])
            else:
                shape_prod = math.prod((*self.minres, self.depths[-1]))
                self.space_linear = LinearLayer(feat_dim, shape_prod)
                self.spacenorm = get_norm(norm, self.depths[-1])

            self.deconv_layers = nn.ModuleList()
            self.deconv_norms = nn.ModuleList()
            for i in reversed(range(len(self.depths) - 1)):
                dep = self.depths[i]
                in_ch = self.depths[i + 1] if i < len(self.depths) - 2 else self.depths[-1]
                # 实际上逆序遍历时 in_ch 需要重新考虑
                pass

            # 重新构建反卷积
            self.deconv_layers = nn.ModuleList()
            self.deconv_norms = nn.ModuleList()
            in_ch = self.depths[-1]
            for i in reversed(range(len(self.depths) - 1)):
                dep = self.depths[i]
                if strided:
                    self.deconv_layers.append(
                        Conv2DLayer(in_ch, dep, kernel, stride=2, transp=True))
                else:
                    self.deconv_layers.append(
                        Conv2DLayer(in_ch, dep, kernel, stride=1))
                self.deconv_norms.append(get_norm(norm, dep))
                in_ch = dep

            # 最终输出
            if outer:
                self.imgout = Conv2DLayer(in_ch, self.imgdep, kernel,
                                          stride=1, outscale=outscale)
            elif strided:
                self.imgout = Conv2DLayer(in_ch, self.imgdep, kernel,
                                          stride=2, transp=True, outscale=outscale)
            else:
                self.imgout = Conv2DLayer(in_ch, self.imgdep, kernel,
                                          stride=1, outscale=outscale)

    def initial(self, batch_size: int, device='cpu'):
        return {}

    def truncate(self, entries):
        return {}

    def forward(self, carry: dict, feat: Dict[str, torch.Tensor],
                reset: torch.Tensor, single: bool = False):
        """
        feat: {deter: (..., D), stoch: (..., S, C), logit: (..., S, C)}
        Returns: carry, entries, recons (dict of tensors)
        """
        bshape = reset.shape
        act_fn = get_act(self._act)()
        K = self._kernel

        stoch = feat['stoch'].reshape(*feat['stoch'].shape[:-2], -1)
        deter = feat['deter']
        # print(f"[DEBUG] feat['stoch'].shape: {feat['stoch'].shape}")
        # print(f"[DEBUG] feat['deter'].shape: {feat['deter'].shape}")
        inp = torch.cat([stoch, deter], dim=-1)
        inp = inp.reshape(-1, inp.shape[-1])  # flatten batch

        recons = {}

        # ── 向量 ──
        if self.veckeys:
            x = self.vec_mlp(inp)
            x = x.reshape(*bshape, *x.shape[1:])

            for k in self.veckeys:
                out = self.vec_heads[k](x)
                recons[k] = out
                # 如果使用 symlog，输出时需要 symexp 还原
                if self._symlog and not getattr(self.obs_space[k], 'discrete', False):
                    recons[k] = recons[k]  # 存储 logits, 在 loss 中处理

        # ── 图像 ──
        if self.imgkeys:
            g = self._bspace
            if g:
                x0 = deter.reshape(-1, deter.shape[-1])
                x1 = stoch.reshape(-1, stoch.shape[-1])
                x0 = self.sp0(x0)  # (B*, u)
                x0 = rearrange(
                    x0, 'b (g h w c) -> b h w (g c)',
                    h=self.minres[0], w=self.minres[1], g=g)
                x1 = act_fn(self.sp1norm(self.sp1(x1)))
                x1 = self.sp2(x1)
                x1 = x1.reshape(-1, self.minres[0], self.minres[1], self.depths[-1])
                # norm on channel dim
                s = x0 + x1  # (B*, H, W, C)
                B_ = s.shape[0]
                s_flat = s.reshape(-1, s.shape[-1])
                s_flat = self.spnorm(s_flat)
                s = act_fn(s_flat.reshape(B_, self.minres[0], self.minres[1], -1))
            else:
                s = self.space_linear(inp)
                s = s.reshape(-1, self.minres[0], self.minres[1], self.depths[-1])
                B_ = s.shape[0]
                s_flat = s.reshape(-1, s.shape[-1])
                s_flat = self.spacenorm(s_flat)
                s = act_fn(s_flat.reshape(B_, self.minres[0], self.minres[1], -1))

            # (B*, H, W, C) → (B*, C, H, W)
            x = s.permute(0, 3, 1, 2)

            for layer, norm in zip(self.deconv_layers, self.deconv_norms):
                if not self._strided:
                    # upsample 2x before conv
                    x = x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
                x = layer(x)
                B_, C_, H_, W_ = x.shape
                x_flat = x.permute(0, 2, 3, 1).reshape(-1, C_)
                x_flat = norm(x_flat)
                x_flat = act_fn(x_flat)
                x = x_flat.reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)

            # 最终输出
            if not self._outer and not self._strided:
                x = x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            x = self.imgout(x)
            x = torch.sigmoid(x)  # (B*, C, H, W)
            x = x.permute(0, 2, 3, 1)  # → (B*, H, W, C)
            x = x.reshape(*bshape, *x.shape[1:])

            split_sizes = [self.obs_space[k].shape[-1] for k in self.imgkeys]
            img_outs = torch.split(x, split_sizes, dim=-1)
            for k, out in zip(self.imgkeys, img_outs):
                recons[k] = out

        return carry, {}, recons