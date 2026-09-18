"""Read original models without writing caches or evaluation outputs there."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path("/workspace/yi/work/shiyan")
sys.path.insert(0,str(ROOT))
import torch
from config import Config
from data.datasets import get_dataloader
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed
from evaluation.quality import evaluate_no_channel,evaluate_ldpc_channel,_image_quality
from communications.ldpc_coding import get_ldpc_code

parser=argparse.ArgumentParser()
parser.add_argument("--model",choices=["raq","independent_rvq"],required=True)
parser.add_argument("--out",required=True)
args=parser.parse_args()
out=Path(args.out)
cfg=Config()
cfg.validate()
setup_seed(42)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
device=torch.device(cfg.DEVICE)
family=("shiyan_raq_src64-64_raq2-64_curriculum_rate044_A_patch_ch256-512" if args.model=="raq" else
        "shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512")
checkpoint=ROOT/"checkpoints"/(family+"_unet2_ds8x2_k64")/"best_vq_deepsc.pth"
raw=torch.load(checkpoint,map_location="cpu",weights_only=False)
checkpoint_meta={k:v for k,v in raw.items() if isinstance(v,(str,int,float,bool,type(None)))}
del raw
model,inferred=build_model_from_checkpoint(str(checkpoint),cfg,device)
loader=get_dataloader(cfg.TEST_DATASET_PATH,batch_size=1,shuffle=False,mode="test",num_workers=0,pin_memory=False)
images=[x.to(device) for x in loader]
report={"model":args.model,"checkpoint":str(checkpoint),"checkpoint_metadata":checkpoint_meta,
        "inferred":inferred,"parameters":sum(p.numel() for p in model.parameters()),
        "dataset":cfg.TEST_DATASET_PATH,"image_files":loader.dataset.image_files,
        "image_shapes":[list(x.shape) for x in images],"gpu":torch.cuda.get_device_name(),
        "seed":42,"results":[]}
def save():
    (out/(args.model+"_diagnostics.json")).write_text(json.dumps(report,ensure_ascii=False,indent=2))
def layout(value):
    model.use_raq=value!="source"
    if value=="source":return
    if args.model=="raq":model.raq_target_list=list(value)
    else:model.independent_raq_rvq_k_lists=[list(x) for x in value]
def run(label,value,snr=None,rate=None,packing="combined"):
    layout(value)
    setup_seed(42)
    start=time.perf_counter()
    if snr is None:
        ms,ps,diag=evaluate_no_channel(model,images,device,return_diagnostics=True)
    else:
        code=get_ldpc_code(int(256*rate),rate=rate)
        nominal=list(value) if args.model=="raq" else [max(x) for x in value]
        ms,ps,diag=evaluate_ldpc_channel(model,images,nominal,snr,code,device,
            modulation="qpsk",return_diagnostics=True,stream_packing=packing)
    record={"label":label,"layout":value,"snr":snr,"ldpc_rate":rate,"packing":packing,
            "psnr":float(ps),"ms_ssim":float(ms),"diagnostics":diag,
            "seconds":time.perf_counter()-start}
    report["results"].append(record);save()
    total=diag.get("total",{})
    print(json.dumps({k:record[k] for k in ["label","layout","psnr","ms_ssim","seconds"]}|
        {"bpp":total.get("payload_bpp"),"cbr":total.get("transmission_ratio"),
         "ber":total.get("ber")}),flush=True)
if args.model=="raq":
    for label,value in [("default_no_channel",[32,16]),("matched_low_no_channel",[8,16]),
                        ("max_no_channel",[64,64]),("source_no_channel","source")]:
        run(label,value)
    run("default_channel",[32,16],snr=6,rate=.75,packing="per_stage")
    run("matched_low_channel",[8,16],snr=3,rate=.5)
else:
    for label,value in [
        ("default_no_channel",[[4,2],[8,2]]),
        ("matched_high_8x4_no_channel",[[8,4],[8,2]]),
        ("matched_high_4x8_no_channel",[[4,8],[8,2]]),
        ("matched_high_16x2_no_channel",[[16,2],[8,2]]),
        ("old_layout_no_channel",[[4,64],[8,2]]),
        ("validation_layout_no_channel",[[16,16],[4,4]]),
        ("source_no_channel","source")]:
        run(label,value)
    layout([[4,2],[8,2]])
    scores=[]
    for image in images:
        encoded=model.forward_test(image)
        indices=[list(x) for x in encoded["indices"]]
        books=[list(x) for x in encoded["codebooks"]]
        indices[1]=indices[1][:1];books[1]=books[1][:1]
        decoded=model.reconstruct_from_indices(indices,feature_shapes=encoded["feature_shapes"],codebooks=books)
        ms,ps=_image_quality(image,decoded);scores.append((float(ms),float(ps)))
    report["drop_scale1_stage2_probe"]={"meaning":"Remove only scale-1 second residual stage in RAM; not a retrained ablation",
        "psnr":sum(x[1] for x in scores)/len(scores),"ms_ssim":sum(x[0] for x in scores)/len(scores)}
    save()
    run("default_channel",[[4,2],[8,2]],snr=3,rate=.5)
    run("matched_high_channel",[[8,4],[8,2]],snr=6,rate=.75)
save()
