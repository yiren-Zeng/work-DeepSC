"""Run independent native MDRQ allocations with the original AWGN link."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',default='first_new_allocations')
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1])
    parser.add_argument('--rerun',action='store_true',help='Archive existing selected outputs before an isolated rerun')
    args=parser.parse_args()
    probe=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    print('GPU STATUS\n'+probe.stdout,flush=True)
    for line in probe.stdout.splitlines():
        index,name,used,total,utilization=[s.strip() for s in line.split(',')]
        if int(index) in args.gpus and (float(used)>1024 or float(utilization)>15):
            raise RuntimeError(f'Requested GPU {index} is already occupied; no experiment started')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    cases=manifest['search_batches'][args.batch]
    assert cases
    def run_worker(index,gpu):
        subset=cases[index::len(args.gpus)]
        if not subset:return
        command=[sys.executable,str(ROOT/'native_experiments.py'),'run','--model','raq_rvq',
                 '--rq-gpu',str(gpu),'--case-id',*subset]
        if args.rerun:command.append('--rerun')
        log=ROOT/f'{args.batch}_gpu{gpu}.log'
        print('GPU',gpu,'NATIVE ALLOCATIONS',len(subset),'LOG',log,flush=True)
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Native search failed: {log}')
        print('GPU',gpu,'BATCH COMPLETE',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(run_worker,i,gpu) for i,gpu in enumerate(args.gpus)]
        for future in concurrent.futures.as_completed(futures):future.result()
    print('BATCH COMPLETE',args.batch,flush=True)


if __name__=='__main__':main()
