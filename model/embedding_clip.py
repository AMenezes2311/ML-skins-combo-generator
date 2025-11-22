import os
import numpy as np
import torch
import clip
from preprocessing import load_all_images

# --------------------------
# STEP 1 – Load CLIP model
# --------------------------

def load_clip_model():
    # Decide device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model + preprocess
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Set model to eval mode
    model.eval()

    return model, preprocess, device


# --------------------------
# STEP 2 – Extract single embedding
# --------------------------

def extract_clip_embedding(model, preprocess, device, image_path):
    # img_tensor should already be preprocessed and on the correct device
    with torch.no_grad():
        # Encode image with CLIP
        features = model.encode_image(image_path)

        # Optional but recommended: L2-normalize the embedding
        features = features / features.norm(dim=-1, keepdim=True)

        # Convert to 1D numpy array
        embedding = features[0].cpu().numpy()

    return embedding


# --------------------------
# STEP 3 – Generate embeddings for a whole folder
# --------------------------

def generate_clip_embeddings(input_folder, output_folder, model, preprocess, device):
    os.makedirs(output_folder, exist_ok=True)

    # Use preprocessing.py to load and preprocess all images
    images = load_all_images(input_folder)

    for skin_id, img_tensor in images.items():
        img_tensor = img_tensor.to(device)
        # Get embedding
        emb = extract_clip_embedding(model, preprocess, device, img_tensor)

        out_path = os.path.join(output_folder, skin_id + ".npy")
        # Save embedding
        np.save(out_path, emb)

    print(f"Done generating CLIP embeddings for {input_folder}")


# --------------------------
# STEP 4 – Main entrypoint
# --------------------------

if __name__ == "__main__":
    model, preprocess, device = load_clip_model()

    # Original cleaned images (after background removal)
    GLOVES_IN = "data_cleaned/gloves"   # or "data/gloves" if you prefer
    KNIVES_IN = "data_cleaned/knives"

    GLOVES_OUT = "embeddings_clip/gloves"
    KNIVES_OUT = "embeddings_clip/knives"

    generate_clip_embeddings(GLOVES_IN, GLOVES_OUT, model, preprocess, device)
    generate_clip_embeddings(KNIVES_IN, KNIVES_OUT, model, preprocess, device)
