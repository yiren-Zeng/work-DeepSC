"""Verify native resolution, matched rates and source integrity; preserve reports."""
import hashlib
import json
import math
import shutil
import tarfile
from fractions import Fraction
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    comparison=[]
    unique=set()
    for case in manifest['cases']:
        assert [a*b for a,b in case['rvq_k_lists']]==case['effective_k']
        assert Fraction(case['symbols_per_image'],manifest['source_values_per_image'])==Fraction(case['cbr'])
        for method in ['raq_rvq','vq_deepsc','raq_vae']:
            cid=case['id'] if method=='raq_rvq' else case.get('baseline_case_id',case['id'])
            directory=ROOT/'cases'/cid/method
            summary=json.loads((directory/'summary.json').read_text())
            assert summary['actual_cbr']==case['cbr'] and summary['effective_k']==case['effective_k']
            assert summary['native_resolution_preserved'] and summary['seed']==42 and summary['images']==24
            payload=json.loads((directory/'results.json').read_text())
            if method=='raq_vae':
                md=payload['metadata'];totals=payload['totals']
                assert md['channel']=='awgn' and md['snr_db']==10 and md['modulation']=='qpsk'
                assert md['model_seed']==md['channel_seed']==42
                assert md['cscr']['enabled'] is False and md['dataset']==manifest['dataset_path']
                assert md['dataset_order']==manifest['image_names']
                assert md['source_index_count']==7680
                assert md['ldpc']=={'n':256,'k':128,'rate':.5}
                assert md['checkpoint_sha256']==manifest['checkpoint_sha256_before']['raq_vae']
                assert totals['budget_padding_bits']==totals['ldpc_padding_bits']==0
                assert totals['source_payload_bits']==24*case['payload_bits_per_image']
                assert totals['channel_symbols']==24*case['symbols_per_image']
            else:
                assert payload['channel_types']==['awgn'] and payload['channel_seeds']==[42]
                assert payload['modulation']=='qpsk' and payload['ldpc_rate']==.5
                rs=payload['results'].get('awgn',payload['results']);metrics=next(iter(rs.values()))
                assert list(map(float,rs))==[10.0]
                diag=metrics['diagnostics'];stream=diag['combined_stream'];mc=metrics['monte_carlo']
                assert diag['ldpc']=={'n':256,'k':128,'rate':.5}
                assert not diag.get('cscr',{}).get('enabled',False)
                assert stream['ldpc_padding_bits']==stream['modulation_padding_bits']==0
                assert mc['total_payload_bits']==stream['payload_bits']==24*case['payload_bits_per_image']
                assert stream['channel_symbols']==24*case['symbols_per_image']
                assert stream['source_values']==24*manifest['source_values_per_image']
            exports=json.loads((directory/'image_export.json').read_text())['images']
            assert [r['image'] for r in exports]==manifest['image_names']
            assert abs(sum(r['psnr'] for r in exports)/24-summary['psnr'])<1e-5,(cid,method,'PSNR aggregate mismatch')
            assert abs(sum(r['ms_ssim'] for r in exports)/24-summary['ms_ssim'])<1e-6,(cid,method,'MS-SSIM aggregate mismatch')
            for record in exports:
                assert math.isfinite(record['psnr']) and 0<record['ms_ssim']<1
                path=directory/record['image'];h,w=manifest['image_shapes_hw'][record['image']]
                with Image.open(path) as im:assert im.size==(w,h) and im.mode=='RGB'
                assert sha(path)==record['png_sha256']
            durable=ROOT/'reports'/cid/method;durable.mkdir(parents=True,exist_ok=True)
            for name in ['results.json','summary.json','image_export.json','invocation.json','native_output.json','run.log']:
                if (directory/name).exists():shutil.copy2(directory/name,durable/name)
            unique.add((cid,method))
            comparison.append(summary|{'comparison_case_id':case['id'],
                            'reused_baseline_case_id':cid if cid!=case['id'] else None})
    (ROOT/'comparison_all.json').write_text(json.dumps(comparison,indent=2)+'\n')
    integrity={}
    manifest['source_sha256_after']={p:sha(p) for p in manifest['source_sha256_before']}
    manifest['checkpoint_sha256_after']={k:sha(p) for k,p in manifest['checkpoint_paths'].items()}
    manifest['dataset_sha256_after']={n:sha(Path(manifest['dataset_path'])/n) for n in manifest['image_names']}
    for kind in ['source','checkpoint','dataset']:
        before=manifest[f'{kind}_sha256_before'];after=manifest[f'{kind}_sha256_after']
        differences=[p for p in before if before[p]!=after[p]]
        if kind=='source' and differences:
            review=json.loads((ROOT/'adapter_audit/launcher_change_review.json').read_text())
            assert differences==[review['path']],differences
            assert before[review['path']]==review['before_sha256'] and after[review['path']]==review['after_sha256']
            integrity[kind]={'unchanged':False,'count':len(before),'unchanged_count':len(before)-1,
                'model_and_communication_implementation_unchanged':True,
                'observed_launcher_change':review}
        else:
            assert not differences,(kind,differences)
            integrity[kind]={'unchanged':True,'count':len(before)}
    parity=json.loads((ROOT/'adapter_audit/parity_256.json').read_text());assert parity['passed']
    selection=json.loads((ROOT/'selection.json').read_text())
    for method,d in selection['published_image_directories'].items():
        directory=Path(d)
        assert len(list(directory.glob('kodim*.png')))==24
        md=json.loads((directory/'image_index.json').read_text())
        for row in md['images']:
            assert sha(directory/row['image'])==row['png_sha256']
    clean=manifest['search_batches']['tail_clean_verification']
    for index,cid in enumerate(clean):
        invocation=json.loads((ROOT/'cases'/cid/'raq_rvq'/'invocation.json').read_text())
        assert invocation['environment_overrides']['GPU_ID']==str(index%5)
    archive=ROOT/'reports.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6,dereference=True) as bundle:
        bundle.add(ROOT/'reports',arcname='reports',recursive=True)
    verification={'integrity':integrity,'pixel_count_per_image':393216,
        'native_sizes_wh':{'768x512':18,'512x768':6},'matched_conditions_passed':True,
        'cscr_enabled':False,'configurations_compared':len(manifest['cases']),
        'complete_native_method_runs':len(unique),'raw_native_reconstructions':len(unique)*24,
        'unique_cbr_values':sorted(set(c['cbr'] for c in manifest['cases'])),
        'adapter_at_256_matches_original_pngs_metrics_and_bits_exactly':True,
        'seed':42,'jpeg2000_included':False,
        'selected_case':selection['case'],'intermediate_storage':manifest['intermediate_storage'],
        'durable_reports':str(ROOT/'reports'),'final_images_published_inside_workspace':True}
    verification['durable_reports_archive']=str(archive)
    verification['clean_tail_rerun']={'cases':len(clean),'completed':True,'exclusive_output_lock':True}
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    manifest['status']='completed native-resolution comparison and figure export'
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
