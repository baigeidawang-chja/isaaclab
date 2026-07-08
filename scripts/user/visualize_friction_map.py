"""Standalone Isaac Sim scene for visualizing a random friction tile map.

Usage:
    ./isaaclab.sh -p scripts/user/visualize_friction_map.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize a random friction map as colored physical tiles.")
parser.add_argument("--seed", type=int, default=7, help="Random seed for the friction field.")
parser.add_argument("--tile_size", type=float, default=0.4, help="Tile side length in meters.")
parser.add_argument("--nx", type=int, default=50, help="Number of tiles along x.")
parser.add_argument("--ny", type=int, default=60, help="Number of tiles along y.")
parser.add_argument("--mu_min", type=float, default=0.2, help="Minimum friction coefficient.")
parser.add_argument("--mu_max", type=float, default=1.0, help="Maximum friction coefficient.")
parser.add_argument("--lowres_nx", type=int, default=12, help="Low-resolution random field cells along x.")
parser.add_argument("--lowres_ny", type=int, default=8, help="Low-resolution random field cells along y.")
parser.add_argument(
    "--dark_bias",
    type=float,
    default=1.8,
    help="Bias friction samples toward lower/darker values. 1.0 is uniform; larger means darker maps.",
)
parser.add_argument(
    "--color_mode",
    type=str,
    default="grayscale",
    choices=["grayscale", "color"],
    help="Use high-contrast grayscale for print or color for presentation.",
)
parser.add_argument("--tile_gap", type=float, default=0.025, help="Small visual gap between tiles for print contrast.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils


def _interpolate_random_field(
    rng: np.random.Generator,
    nx: int,
    ny: int,
    lowres_nx: int,
    lowres_ny: int,
    mu_min: float,
    mu_max: float,
) -> np.ndarray:
    """Create a smooth/blocky random friction field by bilinear interpolation."""
    lowres_nx = max(2, int(lowres_nx))
    lowres_ny = max(2, int(lowres_ny))
    random_unit = rng.uniform(0.0, 1.0, size=(lowres_ny, lowres_nx)) ** float(max(args_cli.dark_bias, 1.0e-6))
    coarse = mu_min + (mu_max - mu_min) * random_unit

    x_coarse = np.linspace(0.0, 1.0, lowres_nx)
    y_coarse = np.linspace(0.0, 1.0, lowres_ny)
    x_full = np.linspace(0.0, 1.0, nx)
    y_full = np.linspace(0.0, 1.0, ny)

    interp_x = np.stack([np.interp(x_full, x_coarse, row) for row in coarse], axis=0)
    field = np.stack([np.interp(y_full, y_coarse, interp_x[:, ix]) for ix in range(nx)], axis=1)
    return np.clip(field, mu_min, mu_max)


def _mu_to_color(mu: float, mu_min: float, mu_max: float) -> tuple[float, float, float]:
    """Map friction to print-friendly grayscale or blue->red colors."""
    t = float(np.clip((mu - mu_min) / max(mu_max - mu_min, 1.0e-6), 0.0, 1.0))
    if args_cli.color_mode == "grayscale":
        # Low friction is dark and high friction is bright. The range avoids
        # washed-out mid tones after screenshot compression and B/W printing.
        value = 0.08 + 0.84 * t
        return (value, value, value)

    stops = np.array(
        [
            [0.05, 0.25, 0.95],
            [0.10, 0.80, 0.35],
            [0.95, 0.90, 0.15],
            [1.00, 0.45, 0.05],
            [0.90, 0.05, 0.02],
        ],
        dtype=np.float32,
    )
    scaled = t * (len(stops) - 1)
    idx = min(int(np.floor(scaled)), len(stops) - 2)
    alpha = scaled - idx
    color = (1.0 - alpha) * stops[idx] + alpha * stops[idx + 1]
    return tuple(float(v) for v in color)


def _spawn_tile(prim_path: str, translation: tuple[float, float, float], size: tuple[float, float, float], mu: float):
    color = _mu_to_color(mu, args_cli.mu_min, args_cli.mu_max)
    tile_cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material_path="visualMaterial",
        physics_material_path="physicsMaterial",
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85, metallic=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(mu),
            dynamic_friction=float(0.8 * mu),
            restitution=0.0,
            friction_combine_mode="min",
            restitution_combine_mode="min",
        ),
    )
    tile_cfg.func(prim_path, tile_cfg, translation=translation)


def _spawn_friction_map():
    rng = np.random.default_rng(args_cli.seed)
    mu_field = _interpolate_random_field(
        rng=rng,
        nx=args_cli.nx,
        ny=args_cli.ny,
        lowres_nx=args_cli.lowres_nx,
        lowres_ny=args_cli.lowres_ny,
        mu_min=args_cli.mu_min,
        mu_max=args_cli.mu_max,
    )

    prim_utils.create_prim("/World/FrictionMap", "Xform")
    tile_size = float(args_cli.tile_size)
    tile_thickness = 0.02
    z_center = -0.5 * tile_thickness
    visual_tile_size = max(0.05, tile_size - float(args_cli.tile_gap))
    x0 = -0.5 * args_cli.nx * tile_size + 0.5 * tile_size
    y0 = -0.5 * args_cli.ny * tile_size + 0.5 * tile_size

    base_cfg = sim_utils.CuboidCfg(
        size=(args_cli.nx * tile_size + 0.08, args_cli.ny * tile_size + 0.08, 0.012),
        visual_material_path="visualMaterial",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.015, 0.015), roughness=0.95),
    )
    base_cfg.func("/World/FrictionMap/dark_base", base_cfg, translation=(0.0, 0.0, -0.028))

    for iy in range(args_cli.ny):
        for ix in range(args_cli.nx):
            x = x0 + ix * tile_size
            y = y0 + iy * tile_size
            mu = float(mu_field[iy, ix])
            _spawn_tile(
                prim_path=f"/World/FrictionMap/tile_{iy:02d}_{ix:02d}",
                translation=(x, y, z_center),
                size=(visual_tile_size, visual_tile_size, tile_thickness),
                mu=mu,
            )

    return mu_field


def _spawn_legend(num_blocks: int = 9):
    prim_utils.create_prim("/World/FrictionLegend", "Xform")
    block_size = 0.35
    gap = 0.08
    z_center = -0.01
    x = 0.5 * args_cli.nx * args_cli.tile_size + 0.75
    y_start = -0.5 * (num_blocks - 1) * (block_size + gap)

    legend_base_cfg = sim_utils.CuboidCfg(
        size=(block_size + 0.08, num_blocks * block_size + (num_blocks - 1) * gap + 0.12, 0.012),
        visual_material_path="visualMaterial",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.015, 0.015), roughness=0.95),
    )
    legend_base_cfg.func("/World/FrictionLegend/dark_base", legend_base_cfg, translation=(x, 0.0, -0.028))

    for idx in range(num_blocks):
        t = idx / max(num_blocks - 1, 1)
        mu = args_cli.mu_min + t * (args_cli.mu_max - args_cli.mu_min)
        y = y_start + idx * (block_size + gap)
        _spawn_tile(
            prim_path=f"/World/FrictionLegend/mu_{idx:02d}",
            translation=(x, y, z_center),
            size=(block_size, block_size, 0.02),
            mu=float(mu),
        )


def _design_scene():
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.92, 1.0))
    light_cfg.func("/World/DomeLight", light_cfg)
    mu_field = _spawn_friction_map()
    _spawn_legend()
    print(
        "[INFO] Friction map generated: "
        f"shape={mu_field.shape}, mu_min={mu_field.min():.3f}, "
        f"mu_max={mu_field.max():.3f}, mu_mean={mu_field.mean():.3f}"
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[24.0, -28.0, 22.0], target=[0.0, 0.0, 0.0])
    _design_scene()
    sim.reset()
    print("[INFO] Setup complete. Use the Isaac Sim viewport to frame and capture the screenshot.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
