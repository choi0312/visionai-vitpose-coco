"""Independent standard-library checks against the saved coordinate-level outputs."""
from pathlib import Path
from collections import Counter
import math, json, os
ROOT=Path(os.environ.get('VISIONAI_ROOT',Path(__file__).resolve().parents[1]))
records=json.loads((ROOT/'results/records.json').read_text())
summary=json.loads((ROOT/'results/metrics.json').read_text())['metrics']
manifest=json.loads((ROOT/'data/manifest.json').read_text())
assert len(set(r['image_id'] for r in records))==100
assert len(records)==284 and len({r['annotation_id'] for r in records})==284
assert sorted({r['image_id'] for r in records})==manifest['image_ids']
assert all(m['license']==4 for m in manifest['images'])
assert all(len(r['gt'])==17 for r in records)
for c in ['clean','degraded','detected']:
    n=0; correct=[0,0]; matched=0; angles=[]
    for r in records:
        p=r[c]; scale=max(r['bbox'][2:])
        matched+=p is not None
        for k,(x,y,v) in enumerate(r['gt']):
            if v==0:continue
            n+=1
            if p:
                px,py=p['keypoints'][k];err=math.hypot(x-px,y-py)/scale
                correct[0]+=err<=.05;correct[1]+=err<=.1
        for triple in [[11,13,15],[12,14,16]]:
            if not p or any(r['gt'][k][2]!=2 for k in triple):continue
            values=[]
            for points in [r['gt'],p['keypoints']]:
                a,b,d=[points[k] for k in triple]
                u=(a[0]-b[0],a[1]-b[1]);v=(d[0]-b[0],d[1]-b[1])
                den=math.hypot(*u)*math.hypot(*v)
                if den<1e-6:break
                values.append(math.degrees(math.acos(max(-1,min(1,(u[0]*v[0]+u[1]*v[1])/den)))))
            if len(values)==2:angles.append(abs(values[0]-values[1]))
    s=summary[c]
    assert n==s['labeled_joints']==3104 and matched==s['matched_persons']
    assert correct==[s['correct05'],s['correct10']]
    assert abs(correct[0]/n-s['pck05'])<1e-12
    assert len(angles)==s['angle_evaluated']
    assert abs(sum(angles)/len(angles)-s['knee_angle_mae_degrees'])<1e-9
    print(c, 'PCK .05',f'{correct[0]}/{n}', 'matched',matched,'angle n',len(angles),'PASS')
print('Coordinate metrics, denominators, image/person IDs, licenses and 2D angles verified.')
