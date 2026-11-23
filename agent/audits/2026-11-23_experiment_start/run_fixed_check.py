"""Audit the unchanged default evaluator's one-environment fixed-command run."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--audit-dir', type=Path, required=True)
args = parser.parse_args()
audit = args.audit_dir.resolve()
if not audit.is_dir() or (audit / 'fixed_check').exists():
    raise RuntimeError('Require an existing new audit directory without a fixed_check run.')
entry = REPO / 'scripts/user/eval_nominal_mpc.py'
sys.argv = [str(entry), '--headless', '--profile', 'fixed', '--speed', '0.6', '--heading', '0.0',
            '--duration', '4', '--seed', '42', '--output_dir', str(audit / 'fixed_check')]


def write(name, data):
    (audit / name).write_text(json.dumps(data, indent=2, default=str) + '\n')


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def cfg_summary(cfg):
    return {'class': type(cfg).__module__ + ':' + type(cfg).__qualname__,
            'action': cfg.actions.throttle_steer.to_dict(),
            'command': cfg.commands.planner_command.to_dict(),
            'sim_dt': cfg.sim.dt, 'decimation': cfg.decimation,
            'control_dt': cfg.sim.dt * cfg.decimation, 'device': cfg.sim.device,
            'num_envs': cfg.scene.num_envs, 'episode_length_s': cfg.episode_length_s,
            'robot_usd': cfg.scene.robot.spawn.usd_path,
            'observation_corruption': cfg.observations.policy.enable_corruption,
            'reset_base': cfg.events.reset_base.params,
            'out_of_bounds': cfg.terminations.out_of_bounds.params}


# Import the existing script without its __main__ block; it launches Isaac Sim
# normally. The gym.make observer records the created environment without changing it.
module = runpy.run_path(str(entry), run_name='baseline_audit_evaluator')
gym = module['gym']
original_make = gym.make
try:
    task = module['args_cli'].task
    default_cfg = module['parse_env_cfg'](task, device=module['args_cli'].device, num_envs=1)
    controller = module['NominalMPC']()
    model_path = controller.config.model_config_path.resolve()
    source_modules = ('isaaclab', 'isaaclab_assets', 'isaaclab_tasks',
        'isaaclab_assets.robots.car_edit',
        'isaaclab_tasks.user.traction_protective_control.mdp.actions',
        'WheeledLab.source.wheeledlab.wheeledlab.envs.mdp.actions.ackermann_actions',
        'isaaclab_tasks.user.traction_protective_control.mpc.osqp_mpc', 'osqp')
    write('runtime_defaults.json', {
        'python': sys.executable, 'argv': sys.argv, 'task_spec': dict(gym.spec(task).kwargs),
        'default_environment': cfg_summary(default_cfg),
        'loaded_modules': {name: file_record(sys.modules[name].__file__) for name in source_modules
                           if name in sys.modules and getattr(sys.modules[name], '__file__', None)},
        'model': file_record(model_path), 'osqp_config': file_record(module['DEFAULT_OSQP_MPC_CONFIG_PATH']),
        'steering_config': file_record(module['DEFAULT_STEERING_CONFIG_PATH']),
        'controller_config': asdict(controller.config),
        'versions': {name: importlib.metadata.version(name) for name in ('isaaclab', 'osqp', 'skrl', 'torch')},
    })

    def observe_make(*positional, **keyword):
        env = original_make(*positional, **keyword)
        base = env.unwrapped
        action = base.action_manager.get_term('throttle_steer')
        robot = base.scene['robot']
        write('runtime_environment.json', {
            'effective_environment': cfg_summary(base.cfg), 'actual_step_dt': base.step_dt,
            'usd': file_record(robot.cfg.spawn.usd_path),
            'action_class_file': file_record(inspect.getfile(type(action))),
            'apply_actions_file': file_record(inspect.getfile(action.apply_actions)),
            'actual_action_scale': action._scale.detach().cpu().tolist(),
            'wheel_ids': action._wheel_ids, 'wheel_names': action._wheel_names,
            'steering_ids': action._steering_ids, 'steering_names': action._steering_names,
            'observation_group_dimensions': base.observation_manager.group_obs_dim,
            'isaac_sim_path': str((REPO / '_isaac_sim').resolve()),
        })
        return env

    gym.make = observe_make
    module['main']()
finally:
    gym.make = original_make
    sys.stdout.flush()
    module['simulation_app'].close()
