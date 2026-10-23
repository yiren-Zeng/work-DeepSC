"""Run the four requested AWGN RVQ configurations through the existing shell entry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[2]
SCRIPT = ROOT/'scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh'
CHECKPOINT = ROOT/'checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64/best_vq_deepsc.pth'
DATASET = Path('/workspace/yi/work/Kodak-256-transform-resize')
CASES = [
    ('case01_k2-4_k2-4', '2,4;2,4', 1),
    ('case02_k8-2_k8-2', '8,2;8,2', 2),
    ('case03_k32-2_k8-8', '32,2;8,8', 3),
    ('case04_k8-64_k8-2', '8,64;8,2', 4),
]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_status(status):
    temporary = OUTPUT/'status.tmp'
    temporary.write_text(json.dumps(status, indent=2)+'\n')
    temporary.replace(OUTPUT/'status.json')


def main():
    with (OUTPUT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, _, _ in CASES:
            assert not (OUTPUT/f'{name}.json').exists(), f'Already evaluated: {name}'
        paths = subprocess.check_output(['git','ls-files','*.py','*.sh'],cwd=ROOT,text=True).splitlines()
        sources = {path:digest(ROOT/path) for path in paths if (ROOT/path).is_file()}
        images = sorted(path for path in DATASET.iterdir() if path.suffix.lower() in ('.png','.jpg','.jpeg'))
        assert len(images)==24
        manifest = {
            'checkpoint':str(CHECKPOINT),'checkpoint_sha256':digest(CHECKPOINT),
            'dataset':str(DATASET),'dataset_sha256':{path.name:digest(path) for path in images},
            'source_sha256':sources,'snrs_db':[3,4.5],'channel':'awgn',
            'ldpc':{'k':160,'n':256,'rate':0.625},'modulation':'qpsk',
            'stream_packing':'combined','channel_seeds':list(range(42,52)),
            'no_resize':True,'cases':[],
        }
        jobs = []
        status = {}
        for name, sizes, gpu in CASES:
            env = os.environ.copy()
            env.update(
                SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS=sizes, SNRS='3 4.5',
                MODULATION='qpsk',LDPC_RATE='0.625',LDPC_N='256',
                CHANNEL_TYPES='awgn',CHANNEL_SEEDS='42 43 44 45 46 47 48 49 50 51',
                NO_CHANNEL='0',GPU_ID=str(gpu),CHECKPOINT=str(CHECKPOINT),
                SIMVQ_TEST_DATASET_PATH=str(DATASET),SIMVQ_TEST_NO_RESIZE='1',
                PYTHONDONTWRITEBYTECODE='1',
            )
            log = (OUTPUT/f'{name}.log').open('ab',buffering=0)
            process = subprocess.Popen(
                ['bash',str(SCRIPT),'--json-output',str(OUTPUT/f'{name}.json')],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                close_fds=True,
            )
            jobs.append((name,process,log))
            status[name]={'state':'running','shell_pid':process.pid,'gpu_id':gpu,'k_lists':sizes}
            manifest['cases'].append({'name':name,'k_lists':sizes,'gpu_id':gpu})
            print(f'Launched {name}: GPU {gpu}, PID {process.pid}',flush=True)
        (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        write_status(status)
        while jobs:
            pending=[]
            for name,process,log in jobs:
                returncode=process.poll()
                if returncode is None:
                    pending.append((name,process,log))
                    continue
                log.close()
                status[name].update(state='completed' if returncode==0 else 'failed',exit_code=returncode)
                write_status(status)
                print(f'{name}: exit={returncode}',flush=True)
            jobs=pending
            if jobs:
                time.sleep(2)
        if any(record['state']!='completed' for record in status.values()):
            sys.exit(1)
        assert digest(CHECKPOINT)==manifest['checkpoint_sha256']
        for path, expected in sources.items():
            assert digest(ROOT/path)==expected,path
        print('All evaluations completed; checkpoint and existing source files unchanged.',flush=True)


if __name__=='__main__':
    main()
