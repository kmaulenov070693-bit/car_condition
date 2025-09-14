# app_gradio.py
import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as T
from models import MultiHeadCarNet


# --- patch for gradio_client boolean schema bug ---
import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type
_orig_json_to_py = _gc_utils._json_schema_to_python_type

def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "any" if schema else "never"
    return _orig_get_type(schema)

def _safe_json_to_py(schema, defs=None):
    if isinstance(schema, bool):
        return "Any" if schema else "Never"
    return _orig_json_to_py(schema, defs)

_gc_utils.get_type = _safe_get_type
_gc_utils._json_schema_to_python_type = _safe_json_to_py
# --- end patch ---

CKPT = "checkpoints/model.pt"  # путь к лучшему чекпоинту

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MultiHeadCarNet(backbone=ckpt.get("backbone", "resnet18"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    size = ckpt.get("img_size", 224)
    tfm = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return model, tfm

model, tfm = load_model(CKPT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(img: Image.Image):
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()
    dirty_prob, damaged_prob = probs
    out = {
        "Чистота (грязный)": float(dirty_prob),
        "Целостность (битый)": float(damaged_prob),
    }
    label_dirty = "ГРЯЗНЫЙ" if dirty_prob >= 0.5 else "ЧИСТЫЙ"
    label_damage = "БИТЫЙ" if damaged_prob >= 0.5 else "ЦЕЛЫЙ"
    label_text = f"{label_dirty} · {label_damage}"
    return out, label_text

with gr.Blocks() as demo:
    gr.Markdown("# Определение состояния автомобиля по фото")
    with gr.Row():
        inp = gr.Image(type="pil", label="Загрузите фото (без номеров)")
        with gr.Column():
            out_probs = gr.Label(num_top_classes=2, label="Вероятности")
            out_text = gr.Textbox(label="Итог", interactive=False)
    btn = gr.Button("Предсказать")
    btn.click(fn=predict, inputs=inp, outputs=[out_probs, out_text])

if __name__ == "__main__":
    demo.launch(show_api=False)




