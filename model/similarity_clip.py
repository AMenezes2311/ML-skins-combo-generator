import os
import numpy as np
from typing import List, Tuple
from combo_loader import load_combos


# --------------------------
# Load all embeddings from a folder
# --------------------------

def load_embeddings(folder: str) -> dict[str, np.ndarray]:
    """
    Load all .npy embeddings from a folder.
    Returns {skin_id: np.array}
    """
    data: dict[str, np.ndarray] = {}

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".npy"):
            continue

        skin_id = filename[:-4]  # strip ".npy"
        path = os.path.join(folder, filename)

        vec = np.load(path)
        data[skin_id] = vec

    return data


# --------------------------
# Cosine similarity
# --------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    vec1, vec2: 1D numpy arrays (any length, e.g. 512 for CLIP)
    """
    dot = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


# --------------------------
# Rank gloves for a given knife
# --------------------------

def rank_gloves_for_knife(
    knife_id: str,
    knife_folder: str,
    glove_folder: str,
    combos_csv: str = "metadata/combos.csv",
    top_k: int = 5,
    direct_boost: float = 0.25,
    indirect_boost_scale: float = 0.15,
) -> List[Tuple[str, float]]:
    """
    direct_boost: added if exact (glove, knife) is in combos
    indirect_boost_scale: multiplied by max similarity to any combo-knife for that glove
    """
    knife_embeds = load_embeddings(knife_folder)
    glove_embeds = load_embeddings(glove_folder)

    combos, glove_to_knives, knife_to_gloves = load_combos(combos_csv)

    if knife_id not in knife_embeds:
        raise KeyError(f"Knife ID '{knife_id}' not found in {knife_folder}")

    query = knife_embeds[knife_id]
    scores: List[Tuple[str, float]] = []

    for glove_id, glove_vec in glove_embeds.items():
        base_sim = cosine_similarity(query, glove_vec)
        score = base_sim

        # 1) Direct combo boost
        if (glove_id, knife_id) in combos:
            score += direct_boost

        # 2) Indirect combo boost via similar knives
        related_knives = glove_to_knives.get(glove_id, [])
        if related_knives:
            sims = []
            for related_knife_id in related_knives:
                rk_vec = knife_embeds.get(related_knife_id)
                if rk_vec is None:
                    continue
                sims.append(cosine_similarity(query, rk_vec))

            if sims:
                max_sim_to_combo_knife = max(sims)
                # e.g. if max_sim is 0.9, indirect bonus ~ 0.9 * 0.15
                score += indirect_boost_scale * max_sim_to_combo_knife

        scores.append((glove_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# --------------------------
# Rank knives for a given glove (reverse lookup)
# --------------------------

def rank_knives_for_glove(
    glove_id: str,
    knife_folder: str,
    glove_folder: str,
    combos_csv: str = "metadata/combos.csv",
    top_k: int = 5,
    boost_value: float = 0.20,
) -> list[tuple[str, float]]:
    """
    Returns a list of (knife_id, score) sorted by descending score.
    Reverse of rank_gloves_for_knife.
    """
    knife_embeds = load_embeddings(knife_folder)
    glove_embeds = load_embeddings(glove_folder)
    combos = load_combos(combos_csv)

    if glove_id not in glove_embeds:
        raise KeyError(f"Glove ID '{glove_id}' not found in {glove_folder}")

    query = glove_embeds[glove_id]
    scores: list[tuple[str, float]] = []

    for knife_id, knife_vec in knife_embeds.items():
        sim = cosine_similarity(query, knife_vec)

        # Boost if this (glove, knife) pair exists in combos metadata
        if (glove_id, knife_id) in combos:
            sim += boost_value

        scores.append((knife_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]