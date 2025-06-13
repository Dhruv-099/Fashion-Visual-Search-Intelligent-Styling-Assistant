# src/utils.py
# Helper functions

import torchvision.transforms as transforms
from PIL import Image

# Define standard image preprocessing used for the CNN model
# These values are standard for models pre-trained on ImageNet
def get_image_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_and_preprocess_image(image_path: str):
    """Loads an image from path and applies standard transformations."""
    try:
        img = Image.open(image_path).convert('RGB')
        preprocess = get_image_transform()
        return preprocess(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path}: {e}")
        return None

# Add other utility functions here as needed
# e.g., price parsing, data validation helpers