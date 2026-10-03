"""Read-only audit of the existing sample layout; never rewrite training indices."""
import argparse
import json
from pathlib import Path


def audit(config_path):
    config = json.loads(config_path.read_text())
    result = []
    for dataset in config['data']['datasets']:
        root = Path(dataset['path'])
        index = root / dataset.get('index', '.index.txt')
        entries = index.read_text().splitlines()
        missing = []
        unsafe = []
        for sample in entries:
            relative = Path(sample)
            if not sample or relative.is_absolute() or '..' in relative.parts:
                unsafe.append(sample)
                continue
            required = ['image.jpg', dataset.get('depth', 'depth.png'), 'meta.json']
            absent = [name for name in required if not (root / relative / name).is_file()]
            if absent:
                missing.append({'sample': sample, 'missing_files': absent})
        result.append({
            'name': dataset['name'], 'root': str(root), 'entries': len(entries),
            'duplicates': len(entries) - len(set(entries)),
            'unsafe_entries': unsafe, 'missing_count': len(missing),
            'missing_samples': missing,
        })
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/train.json')
    parser.add_argument('--output', type=Path, help='Optional JSON audit destination')
    args = parser.parse_args()
    result = audit(args.config)
    for row in result:
        print(f"{row['name']}: entries={row['entries']}, missing={row['missing_count']}, duplicates={row['duplicates']}, unsafe={len(row['unsafe_entries'])}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    raise SystemExit(1 if any(r['missing_count'] or r['unsafe_entries'] or r['duplicates'] for r in result) else 0)
