import os, base64, tempfile
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchxrayvision as xrv
import skimage.io, skimage.transform

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

print('⏳ 加载模型...')
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()
print('✅ 模型加载完成')

def get_gradcam(m, tensor, class_idx):
    gradients, activations = [], []
    def save_grad(grad): gradients.append(grad)
    def fwd_hook(mod, inp, out):
        activations.append(out)
        out.register_hook(save_grad)
    handle = m.features.denseblock4.register_forward_hook(fwd_hook)
    out = m(tensor)
    m.zero_grad()
    out[0, class_idx].backward()
    handle.remove()
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights): cam += w * acts[i]
    cam = np.maximum(cam, 0)
    if cam.max() > 0: cam = (cam - cam.min()) / cam.max()
    return cam

def analyze_image(img_path):
    img = skimage.io.imread(img_path)
    if len(img.shape) == 3: img = img.mean(axis=2)
    img = skimage.transform.resize(img, (224, 224))
    img = xrv.datasets.normalize(img, 255)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)
    with torch.enable_grad():
        preds = model(tensor)
    results = {label: round(float(preds[0, i].item()) * 100, 1)
               for i, label in enumerate(model.pathologies) if label}
    top = sorted([(k, v) for k, v in results.items() if v > 20],
                 key=lambda x: x[1], reverse=True)[:3]
    orig = Image.open(img_path).convert('RGB')
    W, H = orig.size
    colors = ['#FF3B3B', '#FFB800', '#00E676']
    draw = ImageDraw.Draw(orig)
    for idx, (label, prob) in enumerate(top):
        try:
            cidx = list(model.pathologies).index(label)
            t2 = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)
            cam = get_gradcam(model, t2, cidx)
            cam_r = skimage.transform.resize(cam, (H, W))
            hp = np.argwhere(cam_r > cam_r.max() * 0.5)
            if len(hp) > 0:
                cy, cx = int(hp[:, 0].mean()), int(hp[:, 1].mean())
                r = max(min(hp[:, 0].max() - hp[:, 0].min(),
                            hp[:, 1].max() - hp[:, 1].min()) // 2, 20)
                c = colors[idx % len(colors)]
                for t in range(3):
                    draw.rectangle([cx-r-t, cy-r-t, cx+r+t, cy+r+t], outline=c)
                draw.line([(cx-r*1.5, cy), (cx+r*1.5, cy)], fill=c, width=1)
                draw.line([(cx, cy-r*1.5), (cx, cy+r*1.5)], fill=c, width=1)
                lbl = f"{label[:10]} {prob}%"
                lx = cx - r
                ly = cy - r - 18 if cy - r > 20 else cy + r + 4
                draw.rectangle([lx-2, ly-2, lx+len(lbl)*7+4, ly+14], fill=c)
                draw.text((lx+2, ly), lbl, fill='#000')
        except:
            continue
    draw.text((W-150, H-20), 'GradCAM·TorchXRayVision', fill='#888')
    buf = BytesIO()
    orig.save(buf, format='PNG')
    return results, top, 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def index():
    return jsonify({'status': 'ok', 'message': '肺部X光分析系统后端运行中'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'DenseNet121+GradCAM', 'real': True})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': '未收到文件'}), 400
    file = request.files['file']
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ['jpg', 'jpeg', 'png']:
        return jsonify({'error': '请上传 JPG 或 PNG 格式图片'}), 400
    tmp = tempfile.NamedTemporaryFile(suffix='.' + ext, delete=False)
    file.save(tmp.name)
    try:
        results, top, annotated = analyze_image(tmp.name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp.name)
    nodules = [
        {'id': i+1, 'label': k, 'confidence': v,
         'risk_level': '高风险' if v > 60 else '中等风险' if v > 40 else '低风险',
         'suggestion': '建议尽快就诊' if v > 60 else '建议进一步检查' if v > 40 else '建议定期随访'}
        for i, (k, v) in enumerate(top)
    ]
    rk = 'high' if any(v > 60 for _, v in top) else 'medium' if top else 'low'
    return jsonify({
        'status': 'success',
        'model': 'DenseNet121+GradCAM',
        'nodules': nodules,
        'overall': {
            'finding': f'检测到 {len(top)} 项异常' if top else '未见明显异常',
            'risk': rk,
            'highest_confidence': top[0][1] if top else 0
        },
        'annotated_image': annotated,
        'disclaimer': '本结果由真实AI模型生成，仅供研究参考，不构成临床诊断依据。'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
