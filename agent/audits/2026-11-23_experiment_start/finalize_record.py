"""Index immutable audit outputs and check that baseline source/data stayed fixed."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from collect_inventory import CFG, REPO, ROOT, command, fingerprint, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.audit_dir.resolve()
    if (out / 'completion.json').exists():
        raise FileExistsError('Do not overwrite a completed audit record.')
    inventory = read_json(out / 'inventory.json')
    defaults = read_json(out / 'runtime_defaults.json')
    runtime = read_json(out / 'runtime_environment.json')
    metadata = read_json(out / 'fixed_check/metadata.json')
    analysis = read_json(out / 'fixed_check/analysis.json')
    old_files = inventory['source_fingerprints'] + list(inventory['datasets'].values())
    old_files += inventory['identification_artifacts']
    for run in inventory['historical_runs'].values():
        old_files += [run['steps_csv']] + run['files']
    for checkpoint in inventory['historical_ppo']:
        old_files += [checkpoint['checkpoint'], checkpoint['env_config'], checkpoint['agent_config']]
    changed = [f['path'] for f in old_files if f.get('sha256') != fingerprint(f['path']).get('sha256')]
    source_snapshots = out / 'loaded_source_snapshots'
    source_snapshots.mkdir()
    for number, source in enumerate(inventory['source_fingerprints']):
        path = Path(source['path'])
        (source_snapshots / f'{number:03d}_{path.name}').write_bytes(path.read_bytes())

    columns = ('run', 'checkpoint', 'sha256', 'steering_scale', 'wheel_radius', 'wheelbase', 'track_width',
               'sim_dt', 'decimation', 'num_envs', 'configured_training_timesteps', 'speed_range', 'speed_levels',
               'heading_range', 'heading_offsets', 'env_config', 'agent_config')
    with (out / 'historical_ppo_index.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in inventory['historical_ppo']:
            contract = item.get('training_contract', {})
            action, planner = contract.get('action', {}), contract.get('command', {})
            writer.writerow(dict(zip(columns, (
                item['run'], item['checkpoint']['path'], item['checkpoint']['sha256'],
                action.get('scale', [None, None])[1], action.get('wheel_radius'), action.get('base_length'),
                action.get('base_width'), contract.get('sim_dt'), contract.get('decimation'), contract.get('num_envs'),
                item.get('algorithm', {}).get('trainer', {}).get('timesteps'),
                planner.get('desired_speed_range'), planner.get('speed_levels'), planner.get('desired_heading_range'),
                planner.get('heading_offsets'), item['env_config']['path'], item['agent_config']['path'],
            ))))

    # Read the same bundled OSQP path captured during the single simulator run.
    package_root = str(Path(defaults['loaded_modules']['osqp']['path']).parents[1])
    code = ('import sys,json; sys.path.insert(0,sys.argv[1]); import osqp; '
            'print(json.dumps({"file":osqp.__file__,"python_package":osqp.__version__, '
            '"native_solver":osqp.OSQP().version()}))')
    osqp_version = json.loads(subprocess.check_output([sys.executable, '-c', code, package_root], text=True))
    historical = {}
    for name, run in inventory['historical_runs'].items():
        historical[name] = {'rows': run['steps_csv']['rows_excluding_header'],
            'steps_sha256': run['steps_csv']['sha256'],
            'model_matches_current': run['metadata']['model_config_sha256'] == defaults['model']['sha256'],
            'mpc_matches_current': run['metadata']['mpc_config_sha256'] == defaults['osqp_config']['sha256'],
            'steering_matches_current': run['metadata']['steering_config_sha256'] == defaults['steering_config']['sha256'],
            'usd_matches_current': run['metadata']['robot_usd_sha256'] == runtime['usd']['sha256'],
            'recorded_config_path_missing': [value['recorded_path'] for value in run['config_checks'].values()
                                              if not Path(value['recorded_path']).exists()]}
    with (out / 'fixed_check/steps.csv').open() as file:
        rows = list(csv.DictReader(file))
    result = {
        'completed_local': datetime.now().astimezone().isoformat(),
        'baseline_source_or_data_changed_during_audit': changed,
        'osqp_actual_loaded_package': osqp_version,
        'frozen_model_matches_external_fit': inventory['frozen_model']['sha256'] ==
            fingerprint(ROOT / 'traction_protective_control_nominal_identification/20261120_173554/fit/frozen_parameters.json')['sha256'],
        'validation_certificate_matches_external': fingerprint(CFG / 'nominal_car4wd_bidirectional_validation.json')['sha256'] ==
            fingerprint(ROOT / 'traction_protective_control_nominal_identification/20261120_173554/validation/validation_metrics.json')['sha256'],
        'historical_runs': historical,
        'new_run': {'metadata': metadata, 'analysis': analysis,
            'csv_rows': len(rows), 'status_from_csv': dict(Counter(row['solver_status'] for row in rows)),
            'fallbacks_from_csv': sum(int(row['fallback_used']) for row in rows),
            'termination_from_csv': rows[-1]['termination_reason'],
            'new_csv': fingerprint(out / 'fixed_check/steps.csv'),
            'environment_reference_max_error': max(abs(float(row['environment_desired_speed_m_s']) -
                                                   float(row['desired_speed_m_s'])) for row in rows)},
        'git_after': command('git', 'status', '--short', '--branch'),
        'scope': 'one environment run; no refit, weight change, PPO run, checkpoint rewrite, commit, or deletion',
    }
    (out / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest = {str(path.relative_to(out)): fingerprint(path) for path in sorted(out.rglob('*')) if path.is_file()}
    (out / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'changed': changed, 'osqp': osqp_version, 'historical': historical,
                      'new_rows': len(rows), 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
