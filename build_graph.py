# build graph as nested dictionary

f = open("FB15K_EntityTriples.txt", "r")
lines = f.readlines()

d = {}

for line in lines[:10000]:
    entity1, relation, entity2, _, = line.split()
    relation = relation.replace("/", ", ")[2:]
    if entity1 not in d:
        d[entity1] = {}
    d[entity1][entity2] = relation

import json
json.dump(d, open("fb15k_dict.json", "w"))