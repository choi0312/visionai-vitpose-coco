"""Frozen ViTPose inference: oracle, degraded, and detected boxes on COCO val2017.

Run on a fresh CUDA Colab VM. No training, test-time fitting, or threshold search.
"""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib, json, random, time, platform, zipfile, os, importlib.metadata as md
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageEnhance
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection

ROOT = Path(os.environ.get('VISIONAI_ROOT', '/content/visionai_week9'))
DATA, RESULTS = ROOT/'data', ROOT/'results'
DATA.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True)
SEED, N_IMAGES, DET_THRESHOLD, MATCH_IOU = 42, 100, 0.3, 0.5
POSE_ID = 'usyd-community/vitpose-base-simple'
POSE_REV = 'a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45'
DET_ID = 'PekingU/rtdetr_r50vd_coco_o365'
DET_REV = '457857cec8ac28ddede40ecee9eed2beca321af8'
S3 = 'https://s3.amazonaws.com/images.cocodataset.org'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available(), 'A CUDA GPU is required for this experiment.'

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def download(url, path):
    if path.exists(): return
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120); r.raise_for_status()
            tmp = path.with_suffix(path.suffix+'.part'); tmp.write_bytes(r.content); tmp.replace(path)
            return
        except Exception as error:
            last = error; time.sleep(attempt+1)
    raise last

annotation_path = DATA/'person_keypoints_val2017.json'
if not annotation_path.exists():
    archive = DATA/'annotations_trainval2017.zip'
    print('Downloading official COCO annotations', flush=True)
    download(S3+'/annotations/annotations_trainval2017.zip', archive)
    with zipfile.ZipFile(archive) as z:
        annotation_path.write_bytes(z.read('annotations/person_keypoints_val2017.json'))
ann = json.loads(annotation_path.read_text())
by_image = defaultdict(list)
for a in ann['annotations']:
    if not a['iscrowd'] and a['num_keypoints'] > 0 and min(a['bbox'][2:]) > 0:
        by_image[a['image_id']].append(a)
images = {i['id']: i for i in ann['images']}
# A shareable, explicitly scoped validation sample. License 4 is CC BY 2.0.
eligible = sorted(i for i in images if images[i]['license'] == 4 and
    any(a['num_keypoints'] >= 5 and a['bbox'][2]*a['bbox'][3] >= 32**2 for a in by_image[i]))
selected = sorted(random.Random(SEED).sample(eligible, N_IMAGES))
manifest = {'seed': SEED, 'selection': 'random.sample(sorted eligible image IDs), then sorted',
    'eligibility': 'license=4 (CC BY 2.0), at least one noncrowd person with >=5 labeled keypoints and bbox area>=1024',
    'evaluation': 'all noncrowd persons with num_keypoints>0 and positive bbox dimensions in selected images',
    'eligible_image_count': len(eligible), 'image_ids': selected,
    'annotation_sha256': hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
    'annotation_url': S3+'/annotations/annotations_trainval2017.zip',
    'licenses': ann['licenses'], 'images': [images[i] for i in selected],
    'annotations': [a for i in selected for a in sorted(by_image[i], key=lambda x:x['id'])]}
def get_image(i):
    meta = images[i]; path = DATA/meta['file_name']
    download(S3+'/val2017/'+meta['file_name'], path)
    im = Image.open(path); im.verify()
    return meta['file_name'], hashlib.sha256(path.read_bytes()).hexdigest()
with ThreadPoolExecutor(max_workers=8) as pool:
    manifest['image_sha256'] = dict(pool.map(get_image, selected))
write_json(DATA/'manifest.json', manifest)
print('DATA_READY', len(selected), 'images', len(manifest['annotations']), 'persons', flush=True)

pose_processor = AutoProcessor.from_pretrained(POSE_ID, revision=POSE_REV, use_fast=False)
pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_ID, revision=POSE_REV).cuda().eval().requires_grad_(False)
det_processor = AutoProcessor.from_pretrained(DET_ID, revision=DET_REV, use_fast=False)
det_model = RTDetrForObjectDetection.from_pretrained(DET_ID, revision=DET_REV).cuda().eval().requires_grad_(False)
assert pose_model.config.id2label[13] == 'L_Knee' and det_model.config.id2label[0] == 'person'
pose_config = pose_model.config.to_dict()
write_json(RESULTS/'pose_config.json', pose_config)
pose_processor.save_pretrained(RESULTS/'processor')
timings = defaultdict(float)
def sync(): torch.cuda.synchronize()

@torch.inference_mode()
def infer_pose(im, boxes, heatmap=False):
    if len(boxes) == 0: return [], None
    boxes = np.asarray(boxes, dtype=np.float32)
    poses, maps = [], []
    for start in range(0, len(boxes), 8):
        chunk = boxes[start:start+8]
        sync(); t = time.perf_counter()
        inp = pose_processor(im, boxes=[chunk], return_tensors='pt').to('cuda')
        sync(); timings['pose_preprocess_s'] += time.perf_counter()-t
        t = time.perf_counter(); output = pose_model(**inp); sync()
        timings['pose_forward_s'] += time.perf_counter()-t
        t = time.perf_counter()
        result = pose_processor.post_process_pose_estimation(output, boxes=[chunk], threshold=None)[0]
        for p in result:
            labels = np.asarray(p['labels']); assert labels.tolist() == list(range(17))
            xy = np.asarray(p['keypoints']); score = np.asarray(p['scores'])
            assert xy.shape == (17,2) and np.isfinite(xy).all()
            poses.append({'keypoints': xy.tolist(), 'scores': score.tolist()})
        timings['pose_decode_s'] += time.perf_counter()-t
        if heatmap: maps.append(output.heatmaps.detach().float().cpu().numpy())
    return poses, np.concatenate(maps) if heatmap else None

@torch.inference_mode()
def infer_detector(im):
    sync(); t = time.perf_counter()
    inp = det_processor(images=im, return_tensors='pt').to('cuda')
    output = det_model(**inp); sync()
    r = det_processor.post_process_object_detection(output, target_sizes=torch.tensor([(im.height, im.width)]), threshold=DET_THRESHOLD)[0]
    keep = r['labels'] == 0
    boxes = r['boxes'][keep].cpu().numpy(); scores = r['scores'][keep].cpu().numpy()
    boxes[:,0::2] = np.clip(boxes[:,0::2],0,im.width)
    boxes[:,1::2] = np.clip(boxes[:,1::2],0,im.height)
    boxes[:,2:] -= boxes[:,:2]
    ok = (boxes[:,2] > 0) & (boxes[:,3] > 0)
    timings['detector_total_s'] += time.perf_counter()-t
    return boxes[ok], scores[ok]

def iou(a,b):
    a=np.asarray(a); b=np.asarray(b)
    xy=np.maximum(a[:2],b[:2]); end=np.minimum(a[:2]+a[2:],b[:2]+b[2:])
    area=np.prod(np.maximum(end-xy,0))
    return float(area/(a[2]*a[3]+b[2]*b[3]-area+1e-9))

def matches(gt, boxes, scores):
    used=set(); out={}
    # Fixed, score-ordered one-to-one matching; unused/ignored persons are not called false positives.
    for d in np.argsort(-np.asarray(scores)):
        candidates=[(iou(a['bbox'],boxes[d]),g) for g,a in enumerate(gt) if g not in used]
        if not candidates: continue
        overlap,g=max(candidates)
        if overlap >= MATCH_IOU: out[g]=int(d); used.add(g)
    return out

all_records=[]; raw_detections=[]
# A warm-up is excluded from the measured loop.
first=selected[0]; im=Image.open(DATA/images[first]['file_name']).convert('RGB')
infer_pose(im,[by_image[first][0]['bbox']]); infer_detector(im)
timings.clear(); torch.cuda.reset_peak_memory_stats(); wall=time.perf_counter()
for step,i in enumerate(selected):
    im=Image.open(DATA/images[i]['file_name']).convert('RGB')
    gt=sorted(by_image[i],key=lambda a:a['id']); boxes=[a['bbox'] for a in gt]
    clean, hm = infer_pose(im, boxes, heatmap=True)
    degraded_im=ImageEnhance.Brightness(im.filter(ImageFilter.GaussianBlur(radius=2))).enhance(0.5)
    degraded,_=infer_pose(degraded_im,boxes)
    detected_boxes, det_scores=infer_detector(im)
    detected,_=infer_pose(im,detected_boxes)
    mapping=matches(gt,detected_boxes,det_scores)
    raw_detections.append({'image_id':i,'boxes_xywh':detected_boxes.tolist(),'scores':det_scores.tolist(),'poses':detected})
    for g,a in enumerate(gt):
        all_records.append({'image_id':i,'annotation_id':a['id'],'bbox':a['bbox'],'area':a['area'],
            'gt':np.asarray(a['keypoints']).reshape(17,3).tolist(), 'clean':clean[g], 'degraded':degraded[g],
            'detected':detected[mapping[g]] if g in mapping else None,
            'detected_box':detected_boxes[mapping[g]].tolist() if g in mapping else None,
            'detector_score':float(det_scores[mapping[g]]) if g in mapping else None})
    # All heatmaps are small (17x64x48/person), saved to select examples without rerunning the model.
    np.savez_compressed(RESULTS/f'heatmaps_{i}.npz', heatmaps=hm)
    if (step+1)%10==0:print('INFER',step+1,'/',len(selected),'persons',len(all_records),flush=True)
write_json(RESULTS/'records.json',all_records)
write_json(RESULTS/'detections.json',raw_detections)
timings['loop_wall_s']=time.perf_counter()-wall
env={'seed':SEED,'images':len(selected),'persons':len(all_records),'python':platform.python_version(),
    'packages':{k:md.version(k) for k in ['torch','torchvision','transformers','numpy','scipy','Pillow','matplotlib','accelerate']},
    'gpu':torch.cuda.get_device_name(0),'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'dtype':'float32','batch_max':8,
    'pose_id':POSE_ID,'pose_revision':POSE_REV,'detector_id':DET_ID,'detector_revision':DET_REV,
    'pose_parameters':sum(p.numel() for p in pose_model.parameters()),'training_epochs':0,
    'detector_threshold':DET_THRESHOLD,'matching_iou':MATCH_IOU,'timings':dict(timings),
    'degradation':{'GaussianBlur_radius_px':2,'brightness_factor':0.5},
    'evaluation_scope':'COCO val2017 CC BY 2.0 validation subset; pretrained benchmark familiarity possible; not independent deployment validation',
    'training':'Frozen pretrained inference only; no fitting or fine-tuning.'}
write_json(RESULTS/'environment.json',env)
print('INFERENCE_DONE',json.dumps(env),flush=True)
