from similarity import rank_gloves_for_knife

KNIFE_EMB_DIR = "./embeddings/knives"
GLOVE_EMB_DIR = "./embeddings/gloves"

result = rank_gloves_for_knife(
    knife_id="nomad-knife-crimson-web",
    knife_folder=KNIFE_EMB_DIR,
    glove_folder=GLOVE_EMB_DIR,
    top_k=10
)

print("\nTop matches:")
for glove, score in result:
    print(f"  {glove}: {score:.4f}")
