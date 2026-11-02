"""Typeset reformatted/abridged executed code beside saved result excerpts."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
e=json.loads((ROOT/'results/environment.json').read_text())
def font(size,mono=True):
    for name in (['DejaVuSansMono.ttf','/System/Library/Fonts/Menlo.ttc'] if mono else ['DejaVuSans.ttf','/System/Library/Fonts/Helvetica.ttc']):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default(size=size)
im=Image.new('RGB',(1740,540),'white');d=ImageDraw.Draw(im)
d.text((16,8),'Executed code | reformatted and abridged',font=font(30,False),fill='#222222')
d.text((885,8),'Saved environment and metric excerpts',font=font(30,False),fill='#222222')
left=['@torch.inference_mode()', 'def infer_pose(im, boxes, heatmap=False):', '    # ... batching/timing omitted ...', '    inp = pose_processor(', '        im, boxes=[chunk],', "        return_tensors='pt').to('cuda')", '    output = pose_model(**inp)', '    result = (', '        pose_processor', '        .post_process_pose_estimation(', '            output, boxes=[chunk],', '            threshold=None)', '    )[0]']
right=[f"GPU: {e['gpu']} | float32",'Images / people / joints:', '100 / 284 / 3104','PCK .05 original: 86.92% (2698/3104)', 'PCK .05 degraded: 79.45% (2466/3104)', 'Detector matched people: 275/284', '2D knee MAE: 6.09 -> 9.68 deg',f"Loop wall: {e['timings']['loop_wall_s']:.2f} s | epochs: 0"]
for j,t in enumerate(left):d.text((16,59+j*35),t,font=font(29),fill='#24506e')
for j,t in enumerate(right):d.text((885,59+j*53),t,font=font(29),fill='#222222')
d.line([(865,55),(865,525)],fill='#dddddd',width=2)
im.save(ROOT/'figures/code_excerpt.png')
