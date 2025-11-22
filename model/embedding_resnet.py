import os
import numpy as np
import torch
import torchvision.models as models

from preprocessing import load_all_images


# Load ResNet50 model
def load_resnet50():
    # Load pretrained resnet50
    model = models.resnet50(weights="IMAGENET1K_V2")

    # Remove classification head
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Set to eval mode
    model.eval()

    return model

# Extract embeddings
def extract_embedding(model, tensor):
    with torch.no_grad():
        # Pass tensor through the model
        output = model(tensor)

        # Flatten output from [1, 2048, 1, 1] to [2048]
        embedding = output.view(output.size(0), -1).squeeze(0)

        return embedding.numpy()  # Return embedding as numpy array



# Process a whole folder
def generate_embeddings(image_folder, output_folder, model):
    images = load_all_images(image_folder)

    for skin_id, tensor in images.items():
        # Extract embedding using extract_embedding
        embedding = extract_embedding(model, tensor)

        # Create output path
        out_path = os.path.join(output_folder, skin_id + ".npy")
        # Save embedding to .npy file
        np.save(out_path, embedding)

    print("Done generating embeddings for", image_folder)

if __name__ == "__main__":
    # 1. Load the model
    model = load_resnet50()
    print("✅ ResNet50 loaded")

    # 2. Generate glove embeddings
    GLOVE_IMG_DIR = "./data_cleaned/gloves"
    GLOVE_EMB_DIR = "./embeddings/gloves"

    # make sure output folder exists
    os.makedirs(GLOVE_EMB_DIR, exist_ok=True)
    generate_embeddings(GLOVE_IMG_DIR, GLOVE_EMB_DIR, model)
    print("✅ Glove embeddings generated")

    # 3. Generate knife embeddings
    KNIFE_IMG_DIR = "./data_cleaned/knives"
    KNIFE_EMB_DIR = "./embeddings/knives"

    os.makedirs(KNIFE_EMB_DIR, exist_ok=True)
    generate_embeddings(KNIFE_IMG_DIR, KNIFE_EMB_DIR, model)
    print("✅ Knife embeddings generated")

    print("🎉 All done!")
