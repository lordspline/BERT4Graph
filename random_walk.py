# random walk through knowledge graph to generate 5000 "sentences"

import json
d = json.load(open("fb15k_dict.json"))

import random
keys = list(d.keys())
chosen_keys = random.sample(keys, 5000)

sentences = []

for key in chosen_keys:
    current_sentence = [key]
    for i in range(4):
        if key not in d:
            break
        next_key_candidates = list(d[key].keys())
        next_key = random.choice(next_key_candidates)
        relation = d[key][next_key]
        current_sentence.append(relation)
        current_sentence.append(next_key)
        key = next_key
    if len(current_sentence) != 9:
        continue
    sentences.append(current_sentence)

import json
json.dump(sentences, open("sentences.json", "w"))