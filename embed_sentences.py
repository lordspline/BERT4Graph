# generate embeddings for good sentences

from PIL import Image
import requests
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def fetch_image(entity):
    im = Image.open("images/" + entity[3:] + ".png")
    return im

final_out = []
            
for sentence in good_sentences[3:]:
    try:
        curr = []
        print(sentence)
    
        ims = [fetch_image(sentence[i]) for i in range(0, len(sentence), 2)]
        text = [sentence[i] for i in range(1, len(sentence), 2)]

        print(ims[0])
    
        inputs = processor(text=text, images=ims, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        im_embeds = outputs.image_embeds.squeeze()
        text_embeds = outputs.text_embeds.squeeze()
    
        out = torch.zeros(im_embeds.shape[0] + text_embeds.shape[0], im_embeds.shape[1])
        for i in range(text_embeds.shape[0]):
            out[i] = im_embeds[i]
            out[i+1] = text_embeds[i]
        out[-1] = im_embeds[-1]
        print(out.shape)
        final_out.append(out.unsqueeze(0))
    except:
        print("failed")
        continue

final_out = torch.concat(final_out, axis=0)
torch.save(final_out, "sentence_embeddings.torch")