"""Read-only baseline inventory; write new audit evidence, never alter old runs."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml


REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/media/chja/CE54D158C95990271/IsaacLabTrainingData')
REFERENCE = 'db77845751351fe789582db36f7beafec3aaf9a7'
CFG = REPO / 'source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mpc/config'


def command(*args, cwd=REPO):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def fingerprint(path):
    path = Path(path)
    result = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(block)
        result.update(size_bytes=path.stat().st_size, sha256=digest.hexdigest())
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_record(path):
    result = fingerprint(path)
    if result['exists']:
        with Path(path).open(newline='') as file:
            rows = list(csv.DictReader(file))
        result['rows_excluding_header'] = len(rows)
        result['columns'] = list(rows[0]) if rows else []
        for name in ('split', 'solver_status', 'fallback_used', 'terminated', 'truncated', 'termination_reason'):
            if rows and name in rows[0]:
                result[name] = dict(Counter(row[name] for row in rows))
        measured = ('actual_body_vx_m_s', 'actual_equivalent_center_steering_rad',
                    'actual_yaw_rate_rad_s', 'executed_speed_m_s', 'executed_steering_rad')
        result['nonfinite_selected_fields'] = sum(
            not math.isfinite(float(row[key])) for row in rows for key in measured if key in row
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not ROOT.is_dir():
        raise FileNotFoundError(f'External experiment root missing: {ROOT}')
    out.mkdir(parents=True, exist_ok=False)
    record = {'created_local': datetime.now().astimezone().isoformat(), 'reference_commit': REFERENCE,
              'repository': str(REPO), 'python': sys.executable, 'python_version': sys.version}
    record['git'] = {
        'head': command('git', 'rev-parse', 'HEAD'),
        'branch': command('git', 'branch', '--show-current'),
        'status_before': command('git', 'status', '--short', '--branch'),
        'reference': command('git', 'show', '-s', '--format=fuller', REFERENCE),
        'local_origin_main': command('git', 'rev-parse', 'origin/main'),
        'reference_diff': command('git', 'diff', '--stat', REFERENCE, '--', 'source', 'scripts', 'agent', 'AGENTS.md'),
        'cached_diff': command('git', 'diff', '--cached', '--stat'),
        'gitlinks': command('git', 'ls-tree', 'HEAD', 'WheeledLab', 'tdmpc2/tdmpc2'),
        'submodule_status': command('git', 'submodule', 'status'),
        'remote_contacted': False,
    }
    record['subrepositories'] = {}
    for name in ('WheeledLab', 'tdmpc2/tdmpc2'):
        repo = REPO / name
        changed = command('git', 'diff', '--name-only', cwd=repo)['stdout'].splitlines()
        item = {'head': command('git', 'rev-parse', 'HEAD', cwd=repo),
                'branch': command('git', 'branch', '--show-current', cwd=repo),
                'status': command('git', 'status', '--short', cwd=repo),
                'changed_files': [fingerprint(repo / file) for file in changed]}
        diff = command('git', 'diff', '--binary', cwd=repo)['stdout']
        patch = out / (name.replace('/', '_') + '.patch')
        patch.write_text(diff + '\n')
        item['patch'] = fingerprint(patch)
        record['subrepositories'][name] = item

    record['versions'] = {}
    for package in ('isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'osqp', 'skrl', 'torch', 'numpy', 'scipy', 'gymnasium'):
        try:
            record['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record['versions'][package] = None
    sim_version = Path('/home/chja/isaacsim/VERSION')
    record['isaac_sim'] = {'version_file': fingerprint(sim_version),
                           'version': sim_version.read_text().strip() if sim_version.exists() else None,
                           'launcher_link': str((REPO / '_isaac_sim').resolve())}
    record['disk_mount'] = command('findmnt', '-T', str(ROOT), '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS')
    sources = [p for p in (REPO / 'source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control').rglob('*')
               if p.suffix in ('.py', '.json', '.yaml')]
    sources += [REPO / 'scripts/user' / name for name in (
        'eval_nominal_mpc.py', 'analyze_nominal_mpc.py', 'fit_nominal_vehicle_model.py',
        'collect_nominal_vehicle_identification.py', 'command_follow_evaluation.py')]
    sources += [REPO / 'source/isaaclab_assets/isaaclab_assets/robots/car_edit.py',
                REPO / 'WheeledLab/source/wheeledlab/wheeledlab/envs/mdp/actions/ackermann_actions.py']
    record['source_fingerprints'] = [fingerprint(path) for path in sorted(sources)]
    model_path = CFG / 'nominal_car4wd_bidirectional_model.json'
    model = read_json(model_path)
    validation = read_json(CFG / 'nominal_car4wd_bidirectional_validation.json')
    record['frozen_model'] = fingerprint(model_path)
    record['frozen_model']['matches_validation_certificate'] = (
        record['frozen_model']['sha256'] == validation['parameters_sha256'])
    data = ROOT / 'traction_protective_control_nominal_identification/20261120_173554'
    record['datasets'] = {}
    for split in ('identification', 'validation'):
        expected = model['identification']['source_sha256'] if split == 'identification' else validation['validation_sha256']
        declared = model['identification']['source_csv'] if split == 'identification' else validation['validation_csv']
        item = csv_record(data / f'{split}.csv')
        item.update(recorded_path=declared, expected_sha256=expected, hash_matches=item.get('sha256') == expected,
                    path_migrated=str(data / f'{split}.csv') != declared)
        record['datasets'][split] = item
    record['independence'] = {
        'model_validation_used_for_parameter_selection': model['provenance']['validation_used_for_parameter_selection'],
        'parameters_refit_or_selected_on_validation': validation['parameters_refit_or_selected_on_validation'],
        'same_dataset_hash': record['datasets']['identification'].get('sha256') == record['datasets']['validation'].get('sha256'),
    }
    record['identification_artifacts'] = [fingerprint(data / name) for name in (
        'manifest.json', 'protocol.json', 'fit/frozen_parameters.json', 'fit/identification_metrics.json',
        'fit/identification_prediction_curves.png', 'fit/identification_residuals.png',
        'validation/validation_metrics.json', 'validation/validation_prediction_curves.png',
        'validation/validation_residuals.png')]
    record['historical_runs'] = {}
    for name in ('reverse_fixed', 'reverse_heading', 'signed_joint', 'signed_speed'):
        run = ROOT / 'traction_protective_control_nominal_mpc/bidirectional_20261120' / name
        item = {'directory': str(run), 'steps_csv': csv_record(run / 'steps.csv')}
        item['files'] = [fingerprint(run / file) for file in ('metadata.json', 'predictions.jsonl',
            'model_config.json', 'mpc_config.json', 'steering_config.json', 'analysis.json', 'stage_metrics.csv')]
        if (run / 'metadata.json').exists():
            metadata = read_json(run / 'metadata.json')
            item['metadata'] = metadata
            item['config_checks'] = {}
            for label in ('model', 'mpc', 'steering'):
                path = run / f'{label}_config.json'
                actual = fingerprint(path)
                actual.update(recorded_path=metadata[f'{label}_config'],
                              matches_recorded_sha256=actual.get('sha256') == metadata[f'{label}_config_sha256'])
                item['config_checks'][label] = actual
            item['predictions_rows'] = sum(1 for _ in (run / 'predictions.jsonl').open()) if (run / 'predictions.jsonl').exists() else None
        record['historical_runs'][name] = item

    record['historical_ppo'] = []
    roots = sorted(ROOT.glob('traction_protective_control_command_follow*'))
    for parent in roots:
        for checkpoint in sorted(parent.glob('*/checkpoints/best_agent.pt')):
            run = checkpoint.parents[1]
            item = {'run': str(run), 'checkpoint': fingerprint(checkpoint),
                    'all_checkpoint_names': sorted(p.name for p in checkpoint.parent.glob('*.pt'))}
            env_path, agent_path = run / 'params/env.yaml', run / 'params/agent.yaml'
            item['env_config'], item['agent_config'] = fingerprint(env_path), fingerprint(agent_path)
            if env_path.exists():
                env = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
                action = env.get('actions', {}).get('throttle_steer', {})
                planner = env.get('commands', {}).get('planner_command', {})
                item['training_contract'] = {
                    'action': action, 'command': {k: v for k, v in planner.items() if not k.endswith('_cfg')},
                    'sim_dt': env.get('sim', {}).get('dt'), 'decimation': env.get('decimation'),
                    'num_envs': env.get('scene', {}).get('num_envs'), 'episode_length_s': env.get('episode_length_s'),
                    'robot_usd': env.get('scene', {}).get('robot', {}).get('spawn', {}).get('usd_path'),
                    'observation_terms': list(env.get('observations', {}).get('policy', {})),
                    'rewards': env.get('rewards'),
                }
            if agent_path.exists():
                agent = yaml.load(agent_path.read_text(), Loader=yaml.BaseLoader)
                item['algorithm'] = {k: agent.get(k) for k in ('models', 'trainer')}
            record['historical_ppo'].append(item)
    (out / 'inventory.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'datasets': {k: {s: v.get(s) for s in ('rows_excluding_header', 'sha256', 'hash_matches')} for k,v in record['datasets'].items()},
                      'model_hash_matches': record['frozen_model']['matches_validation_certificate'],
                      'ppo_checkpoints': len(record['historical_ppo'])}, indent=2))


if __name__ == '__main__':
    main()
