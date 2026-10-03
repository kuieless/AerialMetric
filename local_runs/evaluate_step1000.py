"""Wait for the current run's step-1000 checkpoint, then run the full benchmark."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

import torch

REPO = Path(__file__).resolve().parents[1]
TRAIN_METADATA = Path('/data1/szq/moge310/moge2-a6000-main/investigation/latest_training_run.json')
TRAIN = json.loads(TRAIN_METADATA.read_text())
CHECKPOINT = Path(TRAIN['workspace']) / 'checkpoint' / '00001000.pt'
OUTPUT = Path('/data1/szq/moge2/benchmark') / (Path(TRAIN['workspace']).name + '-step1000')
STATUS = REPO / 'local_runs' / 'step1000_status.json'
GPU_IDS = '0,1'

def status(state, **extra):
    payload = dict(state=state, updated_at=datetime.datetime.now().isoformat(), checkpoint=str(CHECKPOINT), output=str(OUTPUT), gpu_ids=GPU_IDS, watcher_pid=os.getpid(), **extra)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(payload, ensure_ascii=False), flush=True)

def training_alive():
    try:
        os.kill(TRAIN['pid'], 0)
        return True
    except ProcessLookupError:
        return False

status('waiting_for_checkpoint')
last_size = None
stable = 0
while True:
    size = CHECKPOINT.stat().st_size if CHECKPOINT.exists() else None
    stable = stable + 1 if size and size == last_size else 0
    last_size = size
    if stable >= 3:
        try:
            package = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
            assert isinstance(package.get('model'), dict) and package['model']
            assert package['model_config']['encoder']['backbone'] == 'dinov2_vitl14'
            assert any('lora_A' in k and v.shape[0] == 96 for k, v in package['model'].items())
            del package
            break
        except Exception as exc:
            status('waiting_for_complete_checkpoint', validation_error=str(exc))
    if not training_alive() and not CHECKPOINT.exists():
        status('failed', reason='Training process exited before a valid step-1000 checkpoint was available')
        raise SystemExit(1)
    time.sleep(10)

OUTPUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(TMPDIR='/data1/szq/moge310/tmp', XDG_CACHE_HOME='/data1/szq/moge310/cache', MPLCONFIGDIR='/data1/szq/moge310/cache/matplotlib', HF_HOME='/data1/szq/moge310/cache/huggingface', TORCH_HOME='/data1/szq/moge310/cache/torch', MOGE2_AERIAL=str(CHECKPOINT), OUTPUT_ROOT=str(OUTPUT), PYTHON_BIN='/home/szq/miniconda3/envs/moge310/bin/python', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENCV_IO_ENABLE_OPENEXR='1')
status('running')
with (OUTPUT / 'benchmark.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(['bash', str(REPO / 'benchmark.sh'), GPU_IDS], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    status('running', benchmark_pid=process.pid)
    code = process.wait()
status('complete' if code == 0 else 'failed', exit_code=code, log=str(OUTPUT / 'benchmark.log'))
raise SystemExit(code)
