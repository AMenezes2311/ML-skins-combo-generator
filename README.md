# CS2 Knife/Glove Combo Recommender (ResNet50 + CLIP)

This project generates image embeddings for CS2 knives and gloves, then ranks matching pairs using cosine similarity plus optional boosts from curated combo metadata.

## What it does

Given a **knife**, return the top gloves that match it; given **gloves**, return the top knives.

Ranking is based on:

1. **Cosine similarity** between embeddings
2. **Direct boost** when an exact (glove, knife) pair exists in [`metadata/combos.csv`](metadata/combos.csv)
3. (CLIP only) **Indirect boost** via "glove pairs well with similar knives" logic in [`model.similarity_clip.rank_gloves_for_knife`](model/similarity_clip.py)

## Repo structure (important folders)

- Raw images: [`data/`](data/)
- Background-removed images: [`data_cleaned/`](data_cleaned/)
- ResNet embeddings: [`embeddings/`](embeddings/)
- CLIP embeddings: [`embeddings_clip/`](embeddings_clip/)
- Combo metadata: [`metadata/combos.csv`](metadata/combos.csv)
- Model code: [`model/`](model/)

## Setup

### 1) Create a venv and install deps

```sh
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

## Step 0 (optional): Remove backgrounds

This uses [`remove_background_from_folder`](background_removal.py) to create transparent PNGs in `data_cleaned/`.

```sh
python background_removal.py
```

Inputs/outputs are configured in [`background_removal.py`](background_removal.py).

## ResNet50 pipeline

### 1) Generate embeddings

Run from the `model/` directory (imports are written to work that way):

```sh
cd model
python embedding_resnet.py
```

This writes:

- gloves → `./embeddings/gloves`
- knives → `./embeddings/knives`

See: [`model.embedding_resnet.generate_embeddings`](model/embedding_resnet.py)

### 2) Rank gloves for a knife

Use:

- [`model.similarity_resnet.rank_gloves_for_knife`](model/similarity_resnet.py)

Example runner:

```sh
cd model
python test_resnet_similarity.py
```

## CLIP pipeline

### 1) Generate embeddings

```sh
cd model
python embedding_clip.py
```

This writes:

- gloves → `embeddings_clip/gloves`
- knives → `embeddings_clip/knives`

See: [`model.embedding_clip.generate_clip_embeddings`](model/embedding_clip.py), [`model.embedding_clip.load_clip_model`](model/embedding_clip.py)

### 2) Rank gloves for a knife (with direct + indirect boosts)

Use:

- [`model.similarity_clip.rank_gloves_for_knife`](model/similarity_clip.py)
- [`model.similarity_clip.rank_knives_for_glove`](model/similarity_clip.py)

Example runner:

```sh
cd model
python test_clip_similarity.py
```

Note: if you use CLIP embeddings, make sure you point your test to `./embeddings_clip/...` (not `./embeddings/...`).

## Combo metadata format

[`metadata/combos.csv`](metadata/combos.csv) must include headers:

- `glove_id`
- `knife_id`

Loaded via [`model.combo_loader.load_combos`](model/combo_loader.py).

## Scoring notes

Cosine similarity is:

$$
\cos(\theta) = \frac{\vec{a}\cdot\vec{b}}{\lVert \vec{a} \rVert \lVert \vec{b} \rVert}
$$

Implemented in:

- [`model.similarity_resnet.cosine_similarity`](model/similarity_resnet.py)
- [`model.similarity_clip.cosine_similarity`](model/similarity_clip.py)

CLIP embeddings are already L2-normalized in [`model.embedding_clip.extract_clip_embedding`](model/embedding_clip.py), but cosine similarity is still computed explicitly during ranking.

## Typical workflow (end-to-end)

```sh
# 1) (optional) clean images
python background_removal.py

# 2) embeddings (choose one)
cd model
python embedding_resnet.py
# or
python embedding_clip.py

# 3) ranking tests
python test_resnet_similarity.py
# or
python test_clip_similarity.py
```
