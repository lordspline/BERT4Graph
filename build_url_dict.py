# load image urls into dictionary (util)

f = open("URLS_google.txt")
lines = f.readlines()

import urllib
from PIL import Image

d = {}

i = 0
for line in lines[:]:
    print(i)
    i += 1
    obj = line.replace("\n", "").split("\t")
    entity = "/" + obj[1].split("/")[0].replace(".", "/")
    if entity in d:
        if len(list(d[entity])) >= 20:
            continue
    if entity not in d:
        d[entity] = []
    d[entity].append(obj[0])

import json
json.dump(d, open("urls_dict.json", "w"))