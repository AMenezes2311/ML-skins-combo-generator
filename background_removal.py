import os
from rembg import remove
from PIL import Image

def remove_background_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".jpg", ".png").replace(".jpeg", ".png"))

        print(f"Processing {filename}...")

        # Open input image
        with Image.open(input_path).convert("RGBA") as img:
            # Remove background
            result = remove(img)

            # Save result as PNG (keeps transparency)
            result.save(output_path)

    print(f"Done! Cleaned images saved to: {output_folder}")


if __name__ == "__main__":
    # ORIGINAL FOLDERS
    GLOVES_IN = "data/gloves"
    KNIVES_IN = "data/knives"

    # CLEAN OUTPUT FOLDERS
    GLOVES_OUT = "data_cleaned/gloves"
    KNIVES_OUT = "data_cleaned/knives"

    remove_background_from_folder(GLOVES_IN, GLOVES_OUT)
    remove_background_from_folder(KNIVES_IN, KNIVES_OUT)
