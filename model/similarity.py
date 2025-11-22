import os
import numpy as np
from combo_loader import load_combos

# Load all embeddings from a folder
def load_embeddings(folder):
    """
    Returns {skin_id: np.array}
    """
    data = {}

    for filename in os.listdir(folder):
        # TODO: only load .npy files
        if not filename.endswith(".npy"):
            continue

        skin_id = filename.replace(".npy", "")
        path = os.path.join(folder, filename)

        # Load numpy vector
        vec = np.load(path)

        data[skin_id] = vec  # Store the vector

    return data


# --------------------------
# STEP 2 — Cosine similarity
# --------------------------

def cosine_similarity(vec1, vec2):
    """
    vec1, vec2: np.array (2048,)
    """
    # Dot product
    dot = np.dot(vec1, vec2)

    # Norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Guard against divide-by-zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Return dot / (norm1 * norm2)
    return dot / (norm1 * norm2)


# --------------------------
# STEP 3 — Rank gloves for a given knife
# --------------------------

def rank_gloves_for_knife(knife_id, knife_folder, glove_folder, top_k=10, boost_value=0.20):

    # Load embeddings
    knife_embeds = load_embeddings(knife_folder)
    glove_embeds = load_embeddings(glove_folder)

    # Load combos to find boosts
    combos = load_combos("./metadata/combos.csv")

    # Find query vector
    query = knife_embeds[knife_id]

    scores = []

    # Loop over all gloves and calculate similarity
    for glove_id, glove_vec in glove_embeds.items():
        sim = cosine_similarity(query, glove_vec)

        if (glove_id, knife_id) in combos:
            sim += boost_value

        scores.append((glove_id, sim))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # return top scored gloves
    return scores[:top_k]

