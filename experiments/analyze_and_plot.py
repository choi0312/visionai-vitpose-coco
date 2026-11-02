"""Derive every metric/plot from saved predictions, with explicit denominators."""
from pathlib import Path
from collections import Counter
import json, os, math, csv, hashlib
from urllib.request import urlopen
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance

ROOT=Path(os.environ.get('VISIONAI_ROOT','/content/visionai_week9'))
R=ROOT/'results'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
records=json.loads((R/'records.json').read_text())
manifest=json.loads((ROOT/'data/manifest.json').read_text())
env=json.loads((R/'environment.json').read_text())
cfg=json.loads((R/'pose_config.json').read_text())
meta={i['id']:i for i in manifest['images']}
conds=['clean','degraded','detected']
names={'clean':'Oracle / original','degraded':'Oracle / blur + dark','detected':'RT-DETR / original'}
colors={'clean':'#2878a4','degraded':'#c66b2b','detected':'#389067'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':180,'figure.facecolor':'white'})

def angle(points):
    u=points[0]-points[1]; v=points[2]-points[1]
    den=np.linalg.norm(u)*np.linalg.norm(v)
    return None if den < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(u,v)/den,-1,1))))

def joint_rows(c):
    rows=[]
    for r in records:
        gt=np.asarray(r['gt']); mask=gt[:,2]>0; scale=max(r['bbox'][2:])
        pred=r[c]
        xy=np.asarray(pred['keypoints']) if pred else np.full((17,2),np.nan)
        score=np.asarray(pred['scores']) if pred else np.full(17,-np.inf)
        error=np.linalg.norm(xy-gt[:,:2],axis=1)/scale
        for k in np.where(mask)[0]:
            rows.append({'annotation_id':r['annotation_id'],'joint':int(k),'visibility':int(gt[k,2]),
                'error':float(error[k]) if pred else None,'score':float(score[k]) if pred else None,
                'correct05':bool(pred is not None and error[k]<=0.05),
                'correct10':bool(pred is not None and error[k]<=0.1)})
    return rows

all_joints={c:joint_rows(c) for c in conds}
angle_records=[]
for r in records:
    gt=np.asarray(r['gt'])
    for side,triple in [('left',[11,13,15]),('right',[12,14,16])]:
        if not np.all(gt[triple,2]==2):continue
        actual=angle(gt[triple,:2])
        if actual is None:continue
        a={'annotation_id':r['annotation_id'],'side':side,'gt_degrees':actual}
        for c in conds:
            a[c]=angle(np.asarray(r[c]['keypoints'])[triple]) if r[c] is not None else None
        angle_records.append(a)
metrics={}
for c in conds:
    rows=all_joints[c]; occluded=[r for r in rows if r['visibility']==1]; visible=[r for r in rows if r['visibility']==2]
    paired=[a for a in angle_records if a[c] is not None]
    metrics[c]={'persons':len(records),'matched_persons':sum(r[c] is not None for r in records),
        'labeled_joints':len(rows),'correct05':sum(r['correct05'] for r in rows),'correct10':sum(r['correct10'] for r in rows),
        'pck05':sum(r['correct05'] for r in rows)/len(rows),'pck10':sum(r['correct10'] for r in rows)/len(rows),
        'occluded_joints':len(occluded),'occluded_pck05':sum(r['correct05'] for r in occluded)/len(occluded),
        'visible_joints':len(visible),'visible_pck05':sum(r['correct05'] for r in visible)/len(visible),
        'angle_eligible':len(angle_records),'angle_evaluated':len(paired),
        'knee_angle_mae_degrees':float(np.mean([abs(a[c]-a['gt_degrees']) for a in paired])),
        'knee_angle_median_ae_degrees':float(np.median([abs(a[c]-a['gt_degrees']) for a in paired]))}
confidence=[]
for c in conds:
    for threshold in np.linspace(0,1,21):
        kept=[r for r in all_joints[c] if r['score'] is not None and r['score']>=threshold]
        confidence.append({'condition':c,'threshold':float(threshold),'retained':len(kept),
            'coverage':len(kept)/len(all_joints[c]),
            'conditional_pck05':sum(r['correct05'] for r in kept)/len(kept) if kept else None})
summary={'metrics':metrics,'images':len(manifest['images']),'persons':len(records),
    'definitions':{'PCK_bbox@t':'Euclidean pixel error / max(GT bbox width,height) <= t; micro-average over v>0 joints; unmatched GT person counts as incorrect',
    'visibility':'COCO v=0 unlabeled excluded; v=1 labeled but not visible; v=2 labeled and visible',
    'angles':'arccos dot-product angle hip-knee-ankle in 2D; all three GT joints must have v=2; no confidence filtering; missing detections excluded from MAE and counted in angle coverage',
    'confidence':'raw peak heatmap scores, not calibrated probabilities; conditional accuracy excludes abstained joints; threshold grid is descriptive, not a selected deployment threshold'},
    'eda':{'visibility':dict(Counter(int(k[2]) for r in records for k in r['gt'])),
    'persons_per_image':dict(Counter(r['image_id'] for r in records)),
    'median_bbox_max_side':float(np.median([max(r['bbox'][2:]) for r in records]))}}
def write(name,obj):(R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
write('metrics.json',summary);write('angle_records.json',angle_records);write('confidence_curve.json',confidence)
for c in conds:write(c+'_joint_records.json',all_joints[c])
with (R/'metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['condition',*metrics['clean']]);w.writeheader()
    for c in conds:w.writerow({'condition':c,**metrics[c]})

def save(fig,name):
    fig.savefig(F/name,bbox_inches='tight',pad_inches=.08);plt.close(fig)
def read_image(image_id):
    name=meta[image_id]['file_name']; path=ROOT/'data'/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(urlopen('https://s3.amazonaws.com/images.cocodataset.org/val2017/'+name,timeout=60).read())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['image_sha256'][name]
    return Image.open(path).convert('RGB')
def person_pck(r,c):
    gt=np.asarray(r['gt']); mask=gt[:,2]>0
    return float(np.mean(np.linalg.norm(np.asarray(r[c]['keypoints'])-gt[:,:2],axis=1)[mask]/max(r['bbox'][2:])<=.05)) if r[c] else 0
pool=[r for r in records if np.sum(np.asarray(r['gt'])[:,2]>0)>=12 and r['bbox'][2]>=50 and r['bbox'][3]>=100]
typical=sorted(pool,key=lambda r:(person_pck(r,'clean'),r['annotation_id']))[len(pool)//2]
hard=max(pool,key=lambda r:(person_pck(r,'clean')-person_pck(r,'degraded'),r['annotation_id']))
chosen=[typical,hard] if typical['annotation_id']!=hard['annotation_id'] else [pool[0],hard]
write('selected_examples.json',[{'annotation_id':r['annotation_id'],'image_id':r['image_id'],'clean_pck05':person_pck(r,'clean'),'degraded_pck05':person_pck(r,'degraded')} for r in chosen])

def skeleton(ax,xy,mask,color,alpha=1):
    for a,b in cfg['edges']:
        if mask[a] and mask[b]:ax.plot(xy[[a,b],0],xy[[a,b],1],color=color,lw=1.8,alpha=alpha)
    ax.scatter(xy[mask,0],xy[mask,1],s=11,c=color,edgecolors='white',linewidth=.3,zorder=4,alpha=alpha)
fig,axs=plt.subplots(2,3,figsize=(10,5.6))
for row,r in enumerate(chosen):
    im=read_image(r['image_id']);gt=np.asarray(r['gt'])
    for col,c in enumerate(['gt','clean','degraded']):
        ax=axs[row,col];shown=ImageEnhance.Brightness(im.filter(ImageFilter.GaussianBlur(2))).enhance(.5) if c=='degraded' else im
        ax.imshow(shown); xy=gt[:,:2] if c=='gt' else np.asarray(r[c]['keypoints'])
        # Use identical labeled-joint masks for an interpretable comparison.
        skeleton(ax,xy,gt[:,2]>0,'#35d089' if c=='gt' else '#ffbd29')
        x,y,w,h=r['bbox'];pad=.18
        ax.set_xlim(max(0,x-pad*w),min(im.width,x+(1+pad)*w));ax.set_ylim(min(im.height,y+(1+pad)*h),max(0,y-pad*h));ax.axis('off')
        title='Ground truth' if c=='gt' else f"{names[c]} | PCK .05={person_pck(r,c):.2f}"
        ax.set_title(title,fontsize=10)
    axs[row,0].text(0,-.05,f"image {r['image_id']} / ann {r['annotation_id']}",transform=axs[row,0].transAxes,fontsize=8)
fig.tight_layout(h_pad=1);save(fig,'predictions.png')

fig,axs=plt.subplots(1,3,figsize=(11,2.65))
thresholds=np.arange(.01,.151,.01)
for c in conds:
    errors=np.array([r['error'] if r['error'] is not None else np.inf for r in all_joints[c]])
    axs[0].plot(thresholds,[np.mean(errors<=t)*100 for t in thresholds],label=names[c],color=colors[c])
    cc=[r for r in confidence if r['condition']==c and r['conditional_pck05'] is not None]
    axs[1].plot([r['threshold'] for r in cc],[r['coverage']*100 for r in cc],color=colors[c])
    axs[2].plot([r['coverage']*100 for r in cc],[r['conditional_pck05']*100 for r in cc],color=colors[c])
axs[0].set(xlabel='Tolerance / max bbox side',ylabel='PCK (%)',title='Localization tolerance');axs[0].legend(fontsize=7)
axs[1].set(xlabel='Heatmap score threshold',ylabel='Retained joints (%)',title='Confidence and coverage')
axs[2].set(xlabel='Retained joints (%)',ylabel='PCK .05 of retained (%)',title='Accuracy of retained subset')
for ax in axs:ax.grid(alpha=.2)
fig.tight_layout();save(fig,'condition_curves.png')

fig,axs=plt.subplots(1,3,figsize=(11,2.45))
count=Counter(r['image_id'] for r in records)
axs[0].hist(list(count.values()),bins=np.arange(.5,max(count.values())+1.5,1),color='#2878a4');axs[0].set(xlabel='Labeled people / image',ylabel='Images',title='100-image evaluation sample')
vis=summary['eda']['visibility'];vals=[vis.get(1,0),vis.get(2,0)]
axs[1].bar(['v=1\nnot visible','v=2\nvisible'],vals,color=['#c66b2b','#389067']);axs[1].set(ylabel='Labeled joints',title='Visibility annotations')
for j,n in enumerate(vals):axs[1].text(j,n,str(n),ha='center',va='bottom',fontsize=9)
for j,c in enumerate(conds):
    axs[2].bar(np.arange(2)+(j-1)*.23,[metrics[c]['visible_pck05']*100,metrics[c]['occluded_pck05']*100],width=.23,color=colors[c],label=names[c])
axs[2].set_xticks([0,1],['Visible','Not visible']);axs[2].set(ylabel='PCK .05 (%)',title='Visibility and localization');axs[2].legend(fontsize=6)
fig.tight_layout();save(fig,'eda.png')

# Left panel verifies the angle formula visually; right panel shows actual errors.
fig,axs=plt.subplots(1,2,figsize=(10,2.7))
for c in ['clean','degraded']:
    pairs=[a for a in angle_records if a[c] is not None]
    axs[0].scatter([a['gt_degrees'] for a in pairs],[a[c] for a in pairs],s=11,alpha=.55,label=f"{names[c]} (n={len(pairs)})",color=colors[c])
axs[0].plot([0,180],[0,180],'k--',lw=1);axs[0].set(xlabel='GT projected knee angle (deg)',ylabel='Predicted angle (deg)',title='2D knee angle, visible triplets');axs[0].legend(fontsize=7)
for j,c in enumerate(conds):
    pairs=[a for a in angle_records if a[c] is not None]
    axs[1].bar(j,metrics[c]['knee_angle_mae_degrees'],color=colors[c]);axs[1].text(j,metrics[c]['knee_angle_mae_degrees']+.5,f"n={len(pairs)}",ha='center',fontsize=8)
axs[1].set_xticks(range(3),['Oracle\noriginal','Oracle\nblur + dark','RT-DETR\noriginal']);axs[1].set(ylabel='Absolute error (deg)',title='Angle MAE; unmatched people excluded')
fig.tight_layout();save(fig,'angles.png')

# Real input-space heatmap overlay, using the checkpoint's same affine processor.
from transformers import AutoProcessor
processor=AutoProcessor.from_pretrained(env['pose_id'],revision=env['pose_revision'],use_fast=False)
r=chosen[0];im=read_image(r['image_id'])
img_records=[q for q in records if q['image_id']==r['image_id']];idx=next(i for i,q in enumerate(img_records) if q['annotation_id']==r['annotation_id'])
heatmap=np.load(R/f"heatmaps_{r['image_id']}.npz")['heatmaps'][idx]
crop=processor(im,boxes=[[r['bbox']]],return_tensors='np')['pixel_values'][0].transpose(1,2,0)
crop=np.clip(crop*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
fig,axs=plt.subplots(1,3,figsize=(8,2.9))
for ax,k in zip(axs,[9,13,15]):
    ax.imshow(crop);ax.imshow(heatmap[k],extent=(0,192,256,0),cmap='magma',alpha=.5)
    ax.set_title(cfg['id2label'][str(k)]+' heatmap');ax.axis('off')
fig.tight_layout();save(fig,'heatmaps.png')

# All derived image sources have license 4. Preserve the source URLs and license.
credits=[]
for i in sorted({r['image_id'] for r in chosen}):
    m=meta[i];credits.append({'image_id':i,'file_name':m['file_name'],'source':m.get('flickr_url'),
        'coco_url':m['coco_url'],'license_id':m['license'],'license':'CC BY 2.0','license_url':'https://creativecommons.org/licenses/by/2.0/',
        'changes':'cropped views, skeleton/keypoint overlays; one blur/dark derivative; affine crop and heatmap overlays'})
write('figure_credits.json',credits)
print('METRICS',json.dumps(summary),flush=True)
print('FIGURES_DONE',flush=True)
