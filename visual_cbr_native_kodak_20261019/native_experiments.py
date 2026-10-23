"""Audit native-size accounting, then run unchanged models and physical links."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
STORAGE = Path('/tmp') / ROOT.name
PYTHON = '/home/yi/.conda/envs/work/bin/python'
PROJECTS = {'raq_rvq': BASE/'RAQ-RVQ', 'vq_deepsc': BASE/'VQ-DeepSC', 'raq_vae': BASE/'RAQ-VAE-main'}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command, project, updates, output):
    output.mkdir(parents=True, exist_ok=True)
    write(output/'invocation.json', {'command': command, 'working_directory': str(project),
                                    'environment_overrides': updates})
    env = os.environ | {'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': str(ROOT/'mplconfig')}
    with (output/'run.log').open('w') as stream:
        result = subprocess.run(command, cwd=project, env=env|updates,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Native evaluator failed ({result.returncode}): {output/"run.log"}')


def vae_trial(manifest, case, dataset, central, gpu, adapted):
    if (central/'results.json').exists():
        return json.loads((central/'results.json').read_text())
    project=PROJECTS['raq_vae']
    native=((STORAGE/'vae_trials') if adapted else (project/'runs'))/f'{ROOT.name}_{central.parent.name}_{central.name}_{time.time_ns()}'
    cp=manifest['checkpoint_paths']['raq_vae']
    b,t=case['effective_k']
    updates={'GPU_ID':str(gpu),'CUDA_VISIBLE_DEVICES':str(gpu),
             'CHANNEL_TYPES':'awgn','CHANNEL_SEEDS':'42','SNR':'10','USE_CSCR':'0',
             'BOTTOM_K':str(b),'TOP_K':str(t),'LDPC_RATE':str(case['ldpc_rate']),
             'MODULATION':case['modulation'],'RATIO_DENOMINATOR':str(float(1/Fraction(case['cbr']))),
             'OUTPUT_ROOT':str(native),'DUALK_RUN_DIR':str(Path(cp).parent.parent),
             'PYTHONWARNINGS':'ignore','TF_CPP_MIN_LOG_LEVEL':'3'}
    if adapted:
        evaluator=ROOT/'native_vae_adapter.py'
    else:
        evaluator=project/'eval_seq2seq_ds8_dualk_channel.py'
    command=[PYTHON,'-B',str(evaluator),
                 '--run-dir',updates['DUALK_RUN_DIR'],'--checkpoint',cp,'--dataset',dataset,
                 '--bottom-k',str(b),'--top-k',str(t),'--ldpc-rate',str(case['ldpc_rate']),
                 '--modulation',case['modulation'],'--snr','10','--channel','awgn',
                 '--seed','42','--model-seed','42','--ratio-denominator',updates['RATIO_DENOMINATOR'],
                 '--no-cscr','--concise','--output',str(native)]
    trial=native
    execute(command,project,updates,central)
    payload=json.loads((trial/'results.json').read_text())
    for row in payload['per_image']:
        shutil.copy2(trial/'channel_awgn_10db'/row['image'],central/row['image'])
    shutil.copy2(trial/'results.json',central/'results.json')
    write(central/'native_output.json',{'directory':str(trial),'adapted_accounting':adapted})
    return payload


def audit(manifest, gpu):
    destination=ROOT/'adapter_audit'
    complete=destination/'parity_256.json'
    if complete.exists():
        result=json.loads(complete.read_text());assert result['passed'];print('AUDIT EXISTS',flush=True);return
    case=next(c for c in manifest['cases'] if c['effective_k']==[32,16])
    dataset=str(BASE/'Kodak-256-transform-resize')
    print('AUDIT original VAE, 256x256',flush=True)
    original=vae_trial(manifest,case,dataset,destination/'original256',gpu,False)
    print('AUDIT independent adapter, 256x256',flush=True)
    adapted=vae_trial(manifest,case,dataset,destination/'adapted256',gpu,True)
    assert original['per_image']==adapted['per_image'], 'Adapter changed native per-image metrics/bit counts'
    assert original['metrics']==adapted['metrics'] and original['totals']==adapted['totals']
    pngs=[]
    for row in original['per_image']:
        name=row['image'];a=sha(destination/'original256'/name);b=sha(destination/'adapted256'/name)
        assert a==b,name;pngs.append({'image':name,'identical_png_sha256':a})
    write(complete,{'passed':True,'native_and_adapted_per_image_metrics_identical':True,
                    'native_and_adapted_bit_counts_identical':True,'images':pngs,
                    'original_summary':original['metrics'],
                    'only_differences_allowed':'metadata paths, timestamps and source-count/shape generalization'})
    print('AUDIT PASS: 24 native PNGs, metrics and transmitted bit counts exactly identical',flush=True)


def run_one(manifest,case,method,gpu):
    output=ROOT/'cases'/case['id']/method
    if (output/'summary.json').exists():
        print('EXISTS',case['id'],method,flush=True);return
    print('START',case['id'],method,'GPU',gpu,flush=True)
    if method=='raq_vae':
        payload=vae_trial(manifest,case,manifest['dataset_path'],output,gpu,True)
        md=payload['metadata'];metrics=payload['metrics'];totals=payload['totals']
        assert [md['target_K_bottom'],md['target_K_top']]==case['effective_k']
        assert md['dataset_order']==manifest['image_names'] and md['dataset']==manifest['dataset_path']
        assert md['ldpc']=={'n':case['ldpc_n'],'k':case['ldpc_k'],'rate':case['ldpc_rate']}
        assert md['cscr']['enabled'] is False and md['model_seed']==md['channel_seed']==42
        assert md['source_index_count']==7680
        assert md['channel']=='awgn' and md['snr_db']==10 and md['modulation']==case['modulation']
        assert md['rate_plan']['budget_padding_bits']==0 and totals['budget_padding_bits']==0
        symbols=totals['channel_symbols'];source_values=24*manifest['source_values_per_image'];ber=payload['source_ber']
        exports=[]
        for row in payload['per_image']:
            name=row['image'];exports.append({'image':name,'shape_hw':manifest['image_shapes_hw'][name],
                'psnr':row['psnr'],'ms_ssim':row['ms_ssim'],'png_sha256':sha(output/name)})
        write(output/'image_export.json',{'status':'complete','images':exports,
                                          'original_decoder_and_metrics':True})
    else:
        project=PROJECTS[method]
        updates={'GPU_ID':str(gpu),'CHANNEL_TYPES':'awgn','CHANNEL_SEEDS':'42','SNRS':'10',
                 'MODULATION':case['modulation'],'LDPC_RATE':str(case['ldpc_rate']),
                 'LDPC_N':str(case['ldpc_n']),'NO_CHANNEL':'0','USE_CSCR':'0',
                 'PYTHONPATH':os.pathsep.join([str(ROOT/'hooks'),str(project)]),
                 'NATIVE_EXPORT_ENABLE':'1','NATIVE_EXPORT_PROJECT':str(project),
                 'NATIVE_EXPORT_DATASET':manifest['dataset_path'],
                 'NATIVE_EXPORT_NAMES':str(ROOT/'image_names.json'),
                 'NATIVE_EXPORT_SHAPES':str(ROOT/'image_shapes_hw.json'),
                 'NATIVE_EXPORT_DIR':str(output)}
        if method=='raq_rvq':
            updates.update({'CHECKPOINT':manifest['checkpoint_paths']['raq_rvq'],
                'SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS':';'.join(','.join(map(str,k)) for k in case['rvq_k_lists']),
                'SIMVQ_TEST_DATASET_PATH':manifest['dataset_path'],'SIMVQ_TEST_NO_RESIZE':'1',
                'NATIVE_EXPORT_MAIN':'test_real.py'})
            command=['bash',str(project/'scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh'),
                     '--json-output',str(output/'results.json')]
        else:
            bottom,top=case['effective_k']
            updates.update({'CHECKPOINT':manifest['checkpoint_paths'][f'vq_{bottom}_{top}'],
                            'JSON_OUTPUT':str(output/'results.json'),
                            'NATIVE_EXPORT_MAIN':'test_real_last2_cascade.py'})
            command=['bash',str(project/f'scripts/test/test_k{bottom}_{top}_last2_cascade_200ep.sh')]
        execute(command,project,updates,output)
        payload=json.loads((output/'results.json').read_text())
        if method=='raq_rvq':assert payload['rvq_k_lists']==case['rvq_k_lists']
        else:assert payload['num_embeddings_list']==case['effective_k']
        assert payload['channel_seeds']==[42] and payload['channel_types']==['awgn']
        assert payload['modulation']==case['modulation'] and payload['ldpc_rate']==case['ldpc_rate']
        results=payload['results'].get('awgn',payload['results']);assert list(map(float,results))==[10.0]
        metrics=next(iter(results.values()));mc=metrics['monte_carlo']
        assert not metrics['diagnostics'].get('cscr',{}).get('enabled',False)
        assert mc['num_trials']==1 and mc['channel_seeds']==[42]
        symbols=mc['total_channel_symbols'];source_values=mc['total_source_values'];ber=metrics['ber']
        assert source_values==24*manifest['source_values_per_image']
    actual=Fraction(symbols,source_values)
    assert actual==Fraction(case['cbr'])
    export=json.loads((output/'image_export.json').read_text())
    assert export['status']=='complete' and len(export['images'])==24
    assert [r['image'] for r in export['images']]==manifest['image_names']
    from PIL import Image
    for row in export['images']:
        name=row['image'];height,width=manifest['image_shapes_hw'][name]
        with Image.open(output/name) as image:assert image.mode=='RGB' and image.size==(width,height)
    summary={'case_id':case['id'],'method':method,'display_name':'MDRQ' if method=='raq_rvq' else method,
             'actual_cbr':str(actual),'effective_k':case['effective_k'],'rvq_k_lists':case['rvq_k_lists'],
             'symbols_per_image':symbols//24,'payload_bits_per_image':case['payload_bits_per_image'],
             'images':24,'psnr':metrics['psnr'],'ms_ssim':metrics['ms_ssim'],'ber':ber,
             'native_resolution_preserved':True,'seed':42,'output':str(output)}
    write(output/'summary.json',summary);print('DONE',json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['start','audit','run'])
    parser.add_argument('--model',choices=list(PROJECTS),nargs='+',default=list(PROJECTS))
    parser.add_argument('--case-id',nargs='+')
    parser.add_argument('--audit-gpu',type=int,default=0)
    parser.add_argument('--rq-gpu',type=int,default=0)
    parser.add_argument('--rerun',action='store_true',help='Archive selected completed outputs and recompute them')
    args=parser.parse_args()
    (STORAGE/'cases').mkdir(parents=True,exist_ok=True)
    (STORAGE/'vae_trials').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if args.phase in ['start','audit']:audit(manifest,args.audit_gpu)
    if args.phase=='audit':return
    assert json.loads((ROOT/'adapter_audit/parity_256.json').read_text())['passed']
    cases=[c for c in manifest['cases'] if args.case_id is None or c['id'] in args.case_id];assert cases
    if args.rerun:
        archive=STORAGE/'rerun_archive'/str(time.time_ns())
        for case in cases:
            for method in args.model:
                if method!='raq_rvq' and case.get('baseline_case_id'):continue
                current=ROOT/'cases'/case['id']/method
                if current.exists():
                    destination=archive/case['id']/method
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.move(str(current),str(destination))
    def pipeline(method):
        gpu={'raq_rvq':args.rq_gpu,'vq_deepsc':1,'raq_vae':2}[method]
        for case in cases:
            if method!='raq_rvq' and case.get('baseline_case_id'):
                print('REUSE BASELINE',case['id'],method,case['baseline_case_id'],flush=True)
                continue
            run_one(manifest,case,method,gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(pipeline,method):method for method in args.model}
        failures=[]
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:
                failures.append((futures[future],str(error)));print('ERROR',futures[future],str(error),flush=True)
        if failures:raise RuntimeError(failures)
    print('ALL REQUESTED EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':main()
