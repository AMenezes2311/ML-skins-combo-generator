import csv
from collections import defaultdict

def load_combos(path):
    """
    Returns:
      combos: set of (glove_id, knife_id)
      glove_to_knives: {glove_id: [knife_id, ...]}
      knife_to_gloves: {knife_id: [glove_id, ...]}
    """
    combos = set()
    glove_to_knives = defaultdict(list)
    knife_to_gloves = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            glove_id = row["glove_id"].strip()
            knife_id = row["knife_id"].strip()

            combos.add((glove_id, knife_id))
            glove_to_knives[glove_id].append(knife_id)
            knife_to_gloves[knife_id].append(glove_id)

    return combos, glove_to_knives, knife_to_gloves
