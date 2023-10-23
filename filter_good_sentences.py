# # Filter out good sentences (for which we were able to get all images)

import glob
ims_downloaded = [name[7:-4]for name in glob.glob("images/*")]
print(len(ims_downloaded))

good_sentences = []

for sentence in sentences:
    ims = [sentence[i] for i in range(0, len(sentence), 2)]
    bad = False
    for im in ims:
        if im[3:] not in ims_downloaded:
            bad = True
    if not bad:
        good_sentences.append(sentence)