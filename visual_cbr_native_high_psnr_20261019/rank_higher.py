"""Rank full-image native PSNR without modifying any reconstruction or metric."""
import argparse
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=10)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    rows=[];pairs=[];complete=[]
    for case in manifest['cases']:
        baseline=case.get('baseline_case_id',case['id'])
        dirs={method:ROOT/'cases'/(case['id'] if method=='raq_rvq' else baseline)/method
              for method in ['raq_rvq','vq_deepsc','raq_vae']}
        if not all((p/'summary.json').exists() for p in dirs.values()):continue
        try:
            exports={method:json.loads((p/'image_export.json').read_text()) for method,p in dirs.items()}
        except (FileNotFoundError,json.JSONDecodeError):continue
        if any(e.get('status')!='complete' or len(e.get('images',[]))!=24 for e in exports.values()):continue
        values={method:{row['image']:row for row in report['images']} for method,report in exports.items()}
        if any(set(records)!=set(manifest['image_names']) for records in values.values()):continue
        complete.append(case['id'])
        case_rows=[]
        for name in manifest['image_names']:
            native={method:records[name] for method,records in values.items()}
            scores={method:r['psnr'] for method,r in native.items()}
            gains={method:scores['raq_rvq']-scores[method] for method in ['vq_deepsc','raq_vae']}
            row={'case_id':case['id'],'cbr':case['cbr'],'image':name,
                 'psnr':scores,'ms_ssim':{method:r['ms_ssim'] for method,r in native.items()},
                 'mdrq_psnr':scores['raq_rvq'],'psnr_gains':gains,
                 'mdrq_psnr_above_both':min(gains.values())>0}
            rows.append(row);case_rows.append(row)
        eligible=sorted((r for r in case_rows if r['mdrq_psnr_above_both']),
                        key=lambda r:r['mdrq_psnr'],reverse=True)
        if len(eligible)>=2:
            chosen=eligible[:2]
            pairs.append({'case_id':case['id'],'cbr':case['cbr'],
                          'images':[r['image'] for r in chosen],
                          'minimum_mdrq_psnr':min(r['mdrq_psnr'] for r in chosen),
                          'mean_mdrq_psnr':sum(r['mdrq_psnr'] for r in chosen)/2,
                          'rows':chosen})
    pairs.sort(key=lambda p:(p['minimum_mdrq_psnr'],p['mean_mdrq_psnr']),reverse=True)
    rows.sort(key=lambda r:r['mdrq_psnr'],reverse=True)
    report={'metric_scope':'whole original-resolution image, native evaluator tensor metrics',
            'completed_cases':complete,'planned_cases':len(manifest['cases']),
            'pair_rule':'same case; MDRQ full-image PSNR above both baselines; maximize lower PSNR of two images, then mean',
            'detail_crops_require_separate_visual_inspection':True,'pair_ranking':pairs,
            'all_images_sorted_by_mdrq_psnr':rows}
    (ROOT/'full_psnr_ranking.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',len(complete),'/',len(manifest['cases']))
    for pair in pairs[:args.limit]:
        print(pair['case_id'],pair['images'],[round(r['mdrq_psnr'],5) for r in pair['rows']],
              'minimum gains',[round(min(r['psnr_gains'].values()),5) for r in pair['rows']])


if __name__=='__main__':main()
