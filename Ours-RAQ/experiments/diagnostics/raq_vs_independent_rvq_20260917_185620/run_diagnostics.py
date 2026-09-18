import json,os,subprocess,sys,time
from pathlib import Path
out=Path(__file__).resolve().parent
profiles=json.loads((out/"profiles.json").read_text())
base={k:v for k,v in os.environ.items() if not k.startswith("SIMVQ_")}
jobs=[]
for name,gpu in [("raq","0"),("independent_rvq","1")]:
    env={**base,**profiles[name],"CUDA_VISIBLE_DEVICES":gpu,"GPU_ID":gpu}
    log=(out/(name+"_diagnostics.log")).open("w")
    cmd=["/home/yi/.conda/envs/work/bin/python","-B","-u",str(out/"worker.py"),"--model",name,"--out",str(out)]
    process=subprocess.Popen(cmd,cwd="/workspace/yi/work/shiyan",env=env,stdout=log,stderr=subprocess.STDOUT)
    jobs.append((name,process,log))
    print("Started",name,"PID",process.pid,"GPU",gpu,flush=True)
for name,process,log in jobs:
    rc=process.wait();log.close()
    print("Completed",name,"exit",rc,flush=True)
    if rc:
        print((out/(name+"_diagnostics.log")).read_text()[-4500:],flush=True)
codes={name:p.returncode for name,p,log in jobs}
(out/"completion.json").write_text(json.dumps(codes,indent=2))
sys.exit(0 if all(x==0 for x in codes.values()) else 1)

