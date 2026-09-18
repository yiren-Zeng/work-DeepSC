import json,sys
from pathlib import Path
ROOT=Path('/workspace/yi/work/shiyan')
sys.path.insert(0,str(ROOT))
import torch
from config import Config
from data.datasets import get_dataloader
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed
from evaluation.quality import _image_quality
torch.set_num_threads(2)
torch.set_grad_enabled(False)
setup_seed(42)
out=Path(__file__).resolve().parent
report_path=out/'independent_rvq_diagnostics.json'
report=json.loads(report_path.read_text())
cfg=Config();cfg.validate();device=torch.device(cfg.DEVICE)
model,_=build_model_from_checkpoint(report['checkpoint'],cfg,device)
loader=get_dataloader(cfg.TEST_DATASET_PATH,batch_size=1,shuffle=False,mode='test',num_workers=0,pin_memory=False)
scores=[]
for image in loader:
    image=image.to(device)
    encoded=model.forward_test(image)
    indices=[list(x) for x in encoded['indices']]
    books=[list(x) for x in encoded['codebooks']]
    indices[1]=indices[1][:1];books[1]=books[1][:1]
    decoded=model.reconstruct_from_indices(indices,feature_shapes=encoded['feature_shapes'],codebooks=books)
    ms,ps=_image_quality(image,decoded);scores.append((float(ms),float(ps)))
probe={'meaning':'Remove only scale-1 second residual stage in RAM; not a retrained ablation',
       'psnr':sum(x[1] for x in scores)/len(scores),'ms_ssim':sum(x[0] for x in scores)/len(scores),
       'num_images':len(scores),'grad_disabled':True}
(out/'stage_drop_probe.json').write_text(json.dumps(probe,indent=2))
print(json.dumps(probe))
from evaluation.quality import evaluate_ldpc_channel
from communications.ldpc_coding import get_ldpc_code
model.independent_raq_rvq_k_lists=[[4,64],[8,2]]
setup_seed(42)
ms,ps,diag=evaluate_ldpc_channel(model,loader,[16,4],6,get_ldpc_code(192,rate=.75),device,
    modulation='qpsk',return_diagnostics=True,stream_packing='combined')
old={'label':'old_layout_channel','layout':[[4,64],[8,2]],'snr':6,'ldpc_rate':.75,
     'packing':'combined','psnr':float(ps),'ms_ssim':float(ms),'diagnostics':diag}
(out/'old_layout_channel.json').write_text(json.dumps(old,indent=2))
print(json.dumps({k:v for k,v in old.items() if k!='diagnostics'}))
