import os
from PIL import Image
import torchvision.transforms as transforms

# Transfrom pipeline for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load + preprocess an image
def load_image(path):
    img_tensor = transform(Image.open(path).convert('RGB'))
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

# Load all images in a folder
def load_all_images(folder):
    """Loads all images and returns dict {skin_id: tensor}"""
    data = {}

    for filename in os.listdir(folder):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue

        skin_id = filename.replace(".jpg", "").replace(".png", "")
        path = os.path.join(folder, filename)

        tensor = load_image(path)

        data[skin_id] = tensor

    return data
