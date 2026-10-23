"""Render native whole-image metrics and identical, unenhanced detail crops."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mplconfig'))
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

METHODS=['raq_rvq','vq_deepsc','raq_vae']
PROJECTS={'raq_rvq':'RAQ-RVQ','vq_deepsc':'VQ-DeepSC','raq_vae':'RAQ-VAE-main'}
COLUMNS=[('original','Original'),('vq_deepsc','VQ-DeepSC'),('raq_vae','RAQ-VAE'),('raq_rvq','MDRQ')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection',type=Path,default=ROOT/'selection.json')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--from-published',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    selection=json.loads(args.selection.read_text())
    case=next(c for c in manifest['cases'] if c['id']==selection['case_id'])
    baseline=case.get('baseline_case_id',case['id'])
    directories={m:ROOT/'cases'/(case['id'] if m=='raq_rvq' else baseline)/m for m in METHODS}
    for method,directory in list(directories.items()):
        if args.from_published or not (directory/'summary.json').exists():
            published=selection.get('published_image_directories',{}).get(method)
            if published:directories[method]=Path(published)
    if not all((d/'summary.json').exists() for d in directories.values()):
        raise RuntimeError('Selected native experiment not complete')
    outputs={m:{r['image']:r for r in json.loads((d/'image_export.json').read_text())['images']}
             for m,d in directories.items()}
    destination=ROOT/'inspection' if args.preview else ROOT/'selected'
    destination.mkdir(exist_ok=True)
    metrics=[]
    for sample in selection['images']:
        name=sample['image'];x,y,w,h=sample['xywh']
        original=Path(manifest['dataset_path'])/name
        images={'original':np.asarray(Image.open(original).convert('RGB'))}
        images.update({m:np.asarray(Image.open(d/name).convert('RGB')) for m,d in directories.items()})
        height,width=manifest['image_shapes_hw'][name]
        assert all(image.shape==(height,width,3) for image in images.values())
        assert 0<=x<x+w<=width and 0<=y<y+h<=height
        fig,axes=plt.subplots(2,4,figsize=(14,7),dpi=200)
        fig.subplots_adjust(left=.012,right=.988,bottom=.085,top=.925,wspace=.05,hspace=.04)
        for col,(m,label) in enumerate(COLUMNS):
            image=images[m]
            axes[0,col].imshow(image,interpolation='nearest')
            axes[0,col].add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='#ff2020',lw=1.2))
            axes[0,col].set_title(label,fontsize=13,pad=8)
            axes[1,col].imshow(image[y:y+h,x:x+w],interpolation='nearest')
            axes[0,col].axis('off');axes[1,col].axis('off')
            if m!='original':
                raw=outputs[m][name];score=-10*math.log10(1-raw['ms_ssim'])
                pos=axes[1,col].get_position();center=(pos.x0+pos.x1)/2
                fig.text(center,.035,f'PSNR={raw["psnr"]:.2f} dB | MS-SSIM={score:.2f} dB',
                         ha='center',va='center',fontsize=10)
                mse=np.mean(((image[y:y+h,x:x+w].astype(float)-images['original'][y:y+h,x:x+w])/255)**2)
                metrics.append({'image':name,'method_id':m,'display_name':label,
                    'psnr':raw['psnr'],'ms_ssim':raw['ms_ssim'],'ms_ssim_db':score,
                    'roi_xywh':sample['xywh'],'png_roi_psnr':-10*math.log10(mse),
                    'native_size_wh':[width,height],'png_sha256':raw['png_sha256']})
                assert sha(directories[m]/name)==raw['png_sha256']
        stem=(f'preview_{case["id"]}_{Path(name).stem}' if args.preview else f'{Path(name).stem}_comparison')
        fig.savefig(destination/f'{stem}.png',facecolor='white')
        if not args.preview:fig.savefig(destination/f'{stem}.pdf',facecolor='white')
        plt.close(fig)
    (destination/(f'preview_{case["id"]}_metrics.json' if args.preview else 'comparison_metrics.json')).write_text(
        json.dumps({'case':case,'metric_scope':'whole native image','ms_ssim_db_formula':'-10*log10(1-MS-SSIM)',
                    'crop_interpolation':'nearest; no reconstruction enhancement','metrics':metrics},indent=2)+'\n')
    if args.preview:
        print(json.dumps(metrics,indent=2));return
    published={}
    export_folder=f'high_psnr_native_kodak_20261019_{case["id"]}'
    for m,d in directories.items():
        out=ROOT.parent/PROJECTS[m]/'generated_images'/export_folder/'seed_42'
        out.mkdir(parents=True,exist_ok=True)
        for name in manifest['image_names']:
            if (d/name).resolve()!=(out/name).resolve():shutil.copy2(d/name,out/name)
            assert sha(out/name)==outputs[m][name]['png_sha256']
        for name in ['results.json','summary.json','image_export.json','invocation.json','native_output.json']:
            if (d/name).exists() and (d/name).resolve()!=(out/name).resolve():shutil.copy2(d/name,out/name)
        (out/'image_index.json').write_text(json.dumps({'dataset':manifest['dataset_path'],'case':case,
            'method':m,'seed':42,'native_resolution':True,'images':list(outputs[m].values())},indent=2)+'\n')
        (out/'README.md').write_text(f'Kodak 原始分辨率的全部 24 张原生信道重建图。CBR={case["cbr"]}，'
            f'AWGN、10 dB、QPSK、LDPC 1/2，种子 42，有效码本 K={case["effective_k"]}。'
            f'本次选择 {", ".join(s["image"] for s in selection["images"])} 展示。'
            '未缩放、裁剪或增强原始模型输出。\n')
        published[m]=str(out)
        alias=destination/('mdrq' if m=='raq_rvq' else m)
        if not alias.exists():alias.symlink_to(out,target_is_directory=True)
    original_dir=destination/'original';original_dir.mkdir(exist_ok=True)
    for sample in selection['images']:
        name=sample['image'];shutil.copy2(Path(manifest['dataset_path'])/name,original_dir/name)
    selection.update({'case':case,'metrics':metrics,'published_image_directories':published,
                      'dataset':manifest['dataset_path'],'native_resolution_preserved':True,
                      'layout':'two rows, four columns; full-image metrics only; JPEG-2000 excluded'})
    (ROOT/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    with (destination/'comparison_metrics.csv').open('w') as stream:
        fields=['image','display_name','psnr','ms_ssim','ms_ssim_db','png_roi_psnr']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(metrics)
    print(json.dumps({'case':case['id'],'published':published,'selected':[s['image'] for s in selection['images']]},indent=2))


if __name__=='__main__':main()
