from combo_loader import load_combos

combos = load_combos("../cmetadata/combos.csv")

print(list(combos)[:10])