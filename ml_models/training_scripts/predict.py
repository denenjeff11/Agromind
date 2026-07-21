import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# 1. Classes & Path Configuration
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
MODEL_PATH = "./ml_models/disease_detection/maize_disease_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Check your folder structure!")
        
    model = models.mobilenet_v3_large(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 2. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    model = load_trained_model()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    print("\n--- Prediction Result ---")
    print(f"Disease Detected : {CLASSES[predicted_idx.item()]}")
    print(f"Confidence       : {confidence.item() * 100:.2f}%")
    print("-------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ml_models/training_scripts/predict.py <path_to_leaf_image>")
    else:
        predict_image(sys.argv[1])
