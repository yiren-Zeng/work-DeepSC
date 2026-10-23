"""Rank native reconstruction candidates and render unenhanced common crops."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
METHODS=['raq_rvq','vq_deepsc','raq_vae']


def integral(array):
    return np.pad(np.cumsum(np.cumsum(array,axis=0),axis=1),((1,0),(1,0)))


def area(table,x,y,size):
    return float((table[y+size,x+size]-table[y,x+size]-table[y+size,x]+table[y,x])/(size*size))


def psnr(mse):
    return -10*math.log10(max(mse,1e-15))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--render',type=int,default=0)
    parser.add_argument('--case-id',nargs='+')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    cases=[c for c in manifest['cases'] if args.case_id is None or c['id'] in args.case_id]
    rows=[]
    for case in cases:
        baseline=case.get('baseline_case_id',case['id'])
        directories={m:ROOT/'cases'/(case['id'] if m=='raq_rvq' else baseline)/m for m in METHODS}
        if not all((d/'summary.json').exists() for d in directories.values()):continue
        exports={m:{r['image']:r for r in json.loads((d/'image_export.json').read_text())['images']} for m,d in directories.items()}
        for name in manifest['image_names']:
            native={m:exports[m][name] for m in METHODS}
            images=[np.asarray(Image.open(Path(manifest['dataset_path'])/name),dtype=np.float64)/255]
            images+=[np.asarray(Image.open(directories[m]/name),dtype=np.float64)/255 for m in METHODS]
            reference=images[0];height,width=reference.shape[:2]
            gray=[image@np.array([.299,.587,.114]) for image in images]
            gradients=[np.gradient(image) for image in gray]
            edge=gradients[0][0]**2+gradients[0][1]**2
            edge_integral=integral(edge)
            absolute_gradient=integral(np.sqrt(edge))
            strong_edges=integral((edge>.04**2).astype(float))
            errors=[integral(np.mean((image-reference)**2,axis=2)) for image in images[1:]]
            gradient_errors=[integral((g[0]-gradients[0][0])**2+(g[1]-gradients[0][1])**2) for g in gradients[1:]]
            candidates=[]
            for size in [64,80,96,128]:
                for y in range(0,height-size+1,16):
                    for x in range(0,width-size+1,16):
                        texture=area(absolute_gradient,x,y,size)
                        strong_fraction=area(strong_edges,x,y,size)
                        if texture<.025 or strong_fraction<.06:continue
                        quality=[psnr(area(e,x,y,size)) for e in errors]
                        local_gain=min(quality[0]-q for q in quality[1:])
                        if local_gain<.2:continue
                        energy=area(edge_integral,x,y,size)
                        grad=[area(g,x,y,size) for g in gradient_errors]
                        edge_gain=min((g-grad[0])/max(energy,1e-12) for g in grad[1:])
                        if edge_gain<.06:continue
                        # Favor real edge fidelity over just matching a smooth region's color.
                        score=local_gain+4*edge_gain+.4*math.log2(size/64)
                        candidates.append({'xywh':[x,y,size,size],'png_roi_psnr':quality,
                            'min_psnr_gain':local_gain,'edge_error_normalized_gain':edge_gain,
                            'reference_mean_gradient':texture,'rank_score':score})
            candidates.sort(key=lambda p:p['rank_score'],reverse=True)
            unique=[]
            for patch in candidates:
                x,y,_,_=patch['xywh']
                if any(abs(x-p['xywh'][0])<48 and abs(y-p['xywh'][1])<48 for p in unique):continue
                unique.append(patch)
                if len(unique)==10:break
            values=[native[m]['psnr'] for m in METHODS]
            row={'case_id':case['id'],'cbr':case['cbr'],'image':name,
                 'native_full_psnr':values,'native_full_ms_ssim':[native[m]['ms_ssim'] for m in METHODS],
                 'min_full_psnr_gain':min(values[0]-q for q in values[1:]),
                 'minimum_full_psnr':min(values),'roi_candidates':unique}
            if unique:row['rank_score']=unique[0]['rank_score']+.1*min(values)+.3*row['min_full_psnr_gain']
            else:row['rank_score']=-100
            rows.append(row)
        print('ANALYZED',case['id'],flush=True)
    ranked=sorted(rows,key=lambda r:r['rank_score'],reverse=True)
    (ROOT/'native_diagnostics.json').write_text(json.dumps(ranked,indent=2)+'\n')
    for row in ranked[:30]:
        print(row['case_id'],row['image'],'whole',[round(v,2) for v in row['native_full_psnr']],
              'gain',round(row['min_full_psnr_gain'],2),'roi',row['roi_candidates'][0] if row['roi_candidates'] else None,flush=True)
    output=ROOT/'inspection';output.mkdir(exist_ok=True)
    for row in ranked[:args.render]:
        case=next(c for c in cases if c['id']==row['case_id'])
        baseline=case.get('baseline_case_id',case['id'])
        name=row['image'];roi=row['roi_candidates'][0];x,y,w,h=roi['xywh']
        columns=[('original','Original'),('vq_deepsc','VQ-DeepSC'),('raq_vae','RAQ-VAE'),('raq_rvq','MDRQ')]
        fig,axes=plt.subplots(2,4,figsize=(14,7),dpi=160)
        for i,(m,label) in enumerate(columns):
            path=Path(manifest['dataset_path'])/name if m=='original' else ROOT/'cases'/(case['id'] if m=='raq_rvq' else baseline)/m/name
            image=np.asarray(Image.open(path))
            axes[0,i].imshow(image,interpolation='nearest')
            axes[0,i].add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='red',lw=1))
            title=label if m=='original' else f'{label}: {row["native_full_psnr"][METHODS.index(m)]:.2f} dB'
            axes[0,i].set_title(title)
            axes[1,i].imshow(image[y:y+h,x:x+w],interpolation='nearest')
            axes[0,i].axis('off');axes[1,i].axis('off')
        fig.suptitle(f'{case["id"]} | {name} | ROI {roi["xywh"]}')
        fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.92,hspace=.03,wspace=.04)
        fig.savefig(output/f'{case["id"]}_{Path(name).stem}.png');plt.close(fig)


if __name__=='__main__':main()
