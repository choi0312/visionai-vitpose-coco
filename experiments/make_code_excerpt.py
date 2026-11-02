"""Typeset reformatted/abridged executed code beside saved result excerpts."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
e=json.loads((ROOT/'results/environment.json').read_text())
m=json.loads((ROOT/'results/metrics.json').read_text())['metrics']
def font(size,mono=True):
    for name in (['DejaVuSansMono.ttf','/System/Library/Fonts/Menlo.ttc'] if mono else ['DejaVuSans.ttf','/System/Library/Fonts/Helvetica.ttc']):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default(size=size)
im=Image.new('RGB',(1740,540),'white');d=ImageDraw.Draw(im)
d.text((16,8),'Executed code | reformatted and abridged',font=font(30,False),fill='#222222')
d.text((885,8),'Saved environment and metric excerpts',font=font(30,False),fill='#222222')
left=['@torch.inference_mode()', 'def infer_pose(im, boxes, heatmap=False):', '    # ... batching/timing omitted ...', '    inp = pose_processor(', '        im, boxes=[chunk],', "        return_tensors='pt').to('cuda')", '    output = pose_model(**inp)', '    result = (', '        pose_processor', '        .post_process_pose_estimation(', '            output, boxes=[chunk],', '            threshold=None)', '    )[0]']
right=[f"GPU: {e['gpu']} | float32",'Images / people / joints:',f"{e['images']} / {e['persons']} / {m['clean']['labeled_joints']}",f"PCK .05 original: {m['clean']['pck05']*100:.2f}% ({m['clean']['correct05']}/{m['clean']['labeled_joints']})",f"PCK .05 degraded: {m['degraded']['pck05']*100:.2f}% ({m['degraded']['correct05']}/{m['degraded']['labeled_joints']})",f"Detector matched people: {m['detected']['matched_persons']}/{e['persons']}",f"2D knee MAE: {m['clean']['knee_angle_mae_degrees']:.2f} -> {m['degraded']['knee_angle_mae_degrees']:.2f} deg",f"Loop wall: {e['timings']['loop_wall_s']:.2f} s | epochs: {e['training_epochs']}"]
for j,t in enumerate(left):d.text((16,59+j*35),t,font=font(29),fill='#24506e')
for j,t in enumerate(right):d.text((885,59+j*53),t,font=font(29),fill='#222222')
d.line([(865,55),(865,525)],fill='#dddddd',width=2)
im.save(ROOT/'figures/code_excerpt.png')
