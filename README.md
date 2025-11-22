# Generate CS2 knife/gloves combos
By inputting a knife, the model will return the top 5 gloves that match the knife best. Same applies to gloves, where by giving a pair of gloves, 
the model will return top 5 knives that best match them.

The model ranks the top skins by calculating a score based on the following:
1. Cosine distance between skin 1 and skin 2
2. Combo generated exist in the combos list
3. Combo with different skin and same finish exists in the combo list

# How to run the model
For ResNet50 run files in the following order:
1. Embedding.py
2. Similarity.py
3. test_similarity.py

For CLIP run files in the following order:
1. Clip_embedding_generator.py
2. Similarity_clip.py
3. test_similarity.py
