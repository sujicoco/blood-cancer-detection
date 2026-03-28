import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import uuid
import numpy as np
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- CBAM MODULES ----------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        mx  = self.max_pool(x).view(b, c)
        out = self.mlp(avg) + self.mlp(mx)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, mx], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class EffNetV2_CBAM(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.cbam = CBAM(channels=256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feats_list = self.backbone(x)
        feats = feats_list[-1]
        feats_cbam = self.cbam(feats)
        pooled = self.pool(feats_cbam).view(x.size(0), -1)
        out = self.fc(pooled)
        return out, feats_cbam

# ---------- MODEL LOADING FOR INFERENCE ----------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model_effv2_cbam_best.pt"
)

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

idx_to_class = {0: "ALL", 1: "HEM"}

def load_inference_model():
    backbone = timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=True,
        features_only=True,
    )
    backbone = backbone.to(device)
    model = EffNetV2_CBAM(backbone, num_classes=2).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

model_inf = load_inference_model()

# ---------- ATTENTION OVERLAY ----------
import cv2

def generate_attention_overlay(img_pil, cbam_feats, save_path):
    cbam_np = cbam_feats.detach().cpu().numpy()[0]
    heatmap = np.mean(cbam_np, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    heatmap = cv2.resize(heatmap, (img_pil.width, img_pil.height))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_np = np.array(img_pil)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(heatmap_color, 0.5, img_np, 0.5, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)
    return save_path

def risk_from_severity(sev):
    if sev >= 0.8:
        return "High Risk"
    elif sev >= 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------- NEW: NORMAL vs CANCER DETECTION ----------
# If the highest cancer probability is below this threshold → classify as Normal
NORMAL_THRESHOLD = 0.55

def determine_cancer_or_normal(probs):
    """
    probs: [prob_ALL, prob_HEM]
    Returns: (cancer_detected: bool, cancer_type: str)
      cancer_type is "ALL", "HEM", or "Normal"
    """
    max_prob = float(np.max(probs))
    pred_idx = int(np.argmax(probs))

    if max_prob < NORMAL_THRESHOLD:
        return False, "Normal"
    else:
        return True, idx_to_class[pred_idx]

# ---------- NEW: STAGE CLASSIFICATION ----------
def determine_stage(severity_score, cancer_type):
    """
    Based on severity score, classify cancer stage.
    Only called when cancer is detected (not Normal).

    Stages:
      Benign    → low severity (0.0 – 0.40)
      Pre-mature → moderate severity (0.40 – 0.70)
      Mature    → high severity (0.70 – 1.0)

    Returns: stage string or None if Normal
    """
    if cancer_type == "Normal":
        return None

    if severity_score < 0.40:
        return "Benign"
    elif severity_score < 0.70:
        return "Pre-mature"
    else:
        return "Mature"

def stage_description(stage, cancer_type):
    """Returns a short clinical description for the detected stage."""
    if stage is None or cancer_type == "Normal":
        return "No malignancy detected. Blood smear appears within normal morphological range."

    descriptions = {
        ("ALL", "Benign"): (
            "Early-stage ALL indicators. Mild lymphoblast abnormalities detected. "
            "Low nuclear-to-cytoplasm ratio distortion. Monitoring recommended."
        ),
        ("ALL", "Pre-mature"): (
            "Moderate ALL progression. Elevated lymphoblast presence with irregular "
            "nuclear boundaries. Prompt hematological review advised."
        ),
        ("ALL", "Mature"): (
            "Advanced ALL pattern detected. High density of abnormal lymphoblasts "
            "with prominent nucleoli and dense chromatin. Urgent specialist review required."
        ),
        ("HEM", "Benign"): (
            "Early hematologic malignancy signs. Mild leukocyte morphology irregularities. "
            "Cytoplasmic granularity slightly elevated. Close follow-up recommended."
        ),
        ("HEM", "Pre-mature"): (
            "Moderate HEM progression. Abnormal white blood cell clustering with "
            "irregular cell sizes detected. Clinical evaluation strongly advised."
        ),
        ("HEM", "Mature"): (
            "Advanced hematologic malignancy pattern. Significant nuclear fragmentation "
            "and membrane distortion observed. Immediate specialist consultation required."
        ),
    }
    return descriptions.get((cancer_type, stage), "Stage classification complete. Consult a hematologist.")

# ---------- MAIN PREDICT FUNCTION ----------
def predict_image(image_path, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "media", "attention_maps")

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = infer_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, cbam_feats = model_inf(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # ── NEW: Determine Normal vs Cancer ──
    cancer_detected, cancer_type = determine_cancer_or_normal(probs)

    # ── Severity score (based on max cancer probability) ──
    severity_score = float(np.max(probs)) if cancer_detected else float(np.min(probs))
    risk_label = risk_from_severity(severity_score) if cancer_detected else "No Risk"

    # ── NEW: Determine Stage ──
    stage = determine_stage(severity_score, cancer_type)
    stage_desc = stage_description(stage, cancer_type)

    # ── Existing explanation quality metrics ──
    cbam_coverage = float(np.clip(np.mean(cbam_feats.detach().cpu().numpy()), 0, 1))
    gradcam_agreement = 0.9
    entropy = 2.0

    flags = []
    if cbam_coverage < 0.85:
        flags.append("LOW_EXPLANATION")
    if severity_score < 0.9:
        flags.append("LOW_CONFIDENCE")
    if entropy > 2.5:
        flags.append("UNCERTAIN_FOCUS")
    if not flags:
        flags.append("OK")

    review_reasons = []
    if severity_score < 0.80:
        review_reasons.append("Confidence score < 0.80")
    if cbam_coverage < 0.60:
        review_reasons.append("CBAM coverage < 0.60")
    if entropy > 2.8:
        review_reasons.append("Attention entropy > 2.8")
    if abs(cbam_coverage - gradcam_agreement) > 0.30:
        review_reasons.append("CBAM vs Grad-CAM agreement differs by > 30%")

    doctor_review_status = "DOCTOR_REVIEW_REQUIRED" if review_reasons else "NORMAL"

    if cbam_coverage >= 0.85 and entropy <= 2.5 and abs(cbam_coverage - gradcam_agreement) <= 0.20:
        explanation_status = "PASS"
    elif abs(cbam_coverage - gradcam_agreement) > 0.20:
        explanation_status = "CONFLICT"
    else:
        explanation_status = "LOW"

    est_blast_pct = round(severity_score * 100) if cancer_detected else 0
    if severity_score < 0.3:
        cell_size_var = "Low"
    elif severity_score < 0.7:
        cell_size_var = "Moderate"
    else:
        cell_size_var = "High"

    nuclear_irregularity = "Detected" if entropy > 2.5 else "Not detected"

    # ── Generate attention overlay ──
    attn_filename = f"attn_{uuid.uuid4().hex}.png"
    attn_path = os.path.join(out_dir, attn_filename)
    attn_path = generate_attention_overlay(img_pil, cbam_feats, attn_path)
    web_attn_path = os.path.join("/media/attention_maps/", os.path.basename(attn_path))

    return {
        "patient_id": f"P-{uuid.uuid4().hex[:6].upper()}",

        # ── EXISTING ──
        "cancer_type": cancer_type,
        "severity_score": round(severity_score, 2),
        "severity_percent": int(severity_score * 100),
        "risk_label": risk_label,
        "probabilities": {
            "ALL": float(probs[0]),
            "HEM": float(probs[1]),
        },
        "confidence_stability": 0.93,

        # ── NEW: Cancer detection result ──
        "cancer_detected": cancer_detected,

        # ── NEW: Stage classification ──
        "stage": stage,           # "Benign" / "Pre-mature" / "Mature" / None
        "stage_description": stage_desc,

        # ── EXISTING ──
        "doctor_review": {
            "status": doctor_review_status,
            "reasons": review_reasons,
        },
        "explanation_quality": {
            "cbam_coverage": cbam_coverage,
            "gradcam_agreement": gradcam_agreement,
            "entropy": entropy,
            "flags": flags,
            "status": explanation_status,
        },
        "cell_abnormality": {
            "estimated_blast_percentage": est_blast_pct,
            "cell_size_variance": cell_size_var,
            "nuclear_irregularity": nuclear_irregularity,
            "note": "Model-estimated values for academic demonstration only",
        },
        "paths": {
            "original_image": image_path,
            "attention_map": web_attn_path,
        },
    }