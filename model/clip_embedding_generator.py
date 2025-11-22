import os
import numpy as np
import torch
from PIL import Image
import clip

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
    img = Image.open(image_path).convert("RGB")

    # Preprocess the image and move to device
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode image with CLIP
        features = model.encode_image(img_tensor)

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

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        skin_id = filename.rsplit(".", 1)[0]
        img_path = os.path.join(input_folder, filename)

        # Get embedding
        emb = extract_clip_embedding(model, preprocess, device, img_path)

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
