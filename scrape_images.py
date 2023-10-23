# # Downloading Images (from entities in first 200 sentences)

# load sentences and urls

import json

sentences = json.load(open("sentences.json", "r"))
urls = json.load(open("urls_dict.json", "r"))

import urllib

visited = set()

def fetch_image(entity, urls):
    if entity not in urls:
        return
    if entity in visited:
        return
        
    for url in urls[entity]:
        try:
            urllib.request.urlretrieve(url, "./images/" + entity[3:] + ".png")
            print("yeah")
            visited.add(entity)
            return
        except:
            print("failed")
            continue

i = 0
for sentence in sentences[100:200]:
    print(i)
    i += 1
    ims = [fetch_image(sentence[i], urls) for i in range(0, len(sentence), 2)]