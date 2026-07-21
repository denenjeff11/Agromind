import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Classes & Path Configuration
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
MODEL_PATH = "./ml_models/disease_detection/maize_disease_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Build Model Structure and Load Weights
_model = None

def get_model():
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

# 3. Image Preprocessing (Matches training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def diagnose_leaf(image_path: str) -> dict:
    """
    Takes an image file path, runs inference using maize_disease_model.pth,
    and returns a structured result dictionary.
    """
    model = get_model()
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = CLASSES[predicted_idx.item()]
    confidence_score = float(confidence.item() * 100)
    
    return {
        "disease": predicted_class,
        "confidence": round(confidence_score, 2),
        "is_healthy": predicted_class == "Healthy"
    }
