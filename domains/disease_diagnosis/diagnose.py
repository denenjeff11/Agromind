
"""
Disease diagnosis logic - loads the trained MobileNetV3 model and runs inference.
"""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Raw labels the model was trained on, in this exact order
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Maps raw model output to treatments.json keys
KEY_MAP = {
    'Blight': 'leaf_blight',
    'Common_Rust': 'common_rust',
    'Gray_Leaf_Spot': 'leaf_spot',
    'Healthy': 'healthy',
}

MODEL_PATH = "./ml_models/disease_detection/maize_disease_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_model():
    """Loads the model once, on first use, and caches it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights file not found at: {MODEL_PATH}")

        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        _model = model
    return _model


def diagnose_leaf(image_path: str) -> dict:
    """
    Takes an image file path, runs inference, and returns a structured result.
    """
    model = get_model()

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = CLASSES[predicted_idx.item()]
    confidence_score = float(confidence.item())

    return {
        "disease_key": KEY_MAP[predicted_class],
        "disease_name": predicted_class,
        "confidence": round(confidence_score, 2),
        "is_healthy": predicted_class == "Healthy",
        "crop": "maize",
    }
