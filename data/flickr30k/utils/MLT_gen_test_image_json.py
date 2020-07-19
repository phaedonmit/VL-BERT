"""
Author: Faidon Mitzalis
Generates json file for test set with ambiguous word/caption/images/frcnn for multi30k dataset
"""

words_en = []
words_de = []
captions_en = []
captions_de = []
urls = []
images = []
frcnns = []

with open('../MLT/MLTD_ende_test2016_Faidon.txt') as fp:
    for cnt, line in enumerate(fp):
        line_array = (line.split('|'))
        word_en = line_array[0].strip()
        word_de = line_array[1].strip()
        caption_en = line_array[2].strip()
        caption_de = line_array[3].strip()
        # TODO: read image_id from file!!
        # image_id = '528517845'
        image_id = line_array[4].strip()
        image_id = (image_id.split('.')[0])
        images.append('test_image.zip@/'+ image_id +'.jpg')
        while len(image_id)<8:
            image_id = '0'+image_id
        frcnns.append('test_frcnn.zip@/'+ image_id +'.json')
        words_en.append(word_en)
        words_de.append(word_de)
        captions_en.append(caption_en)
        captions_de.append(caption_de)

# Check lengths are the same  
assert(len(captions_en)==len(captions_de))

import json
with open('../test_MLT_frcnn.json', 'w') as outfile:
    for cnt, (word_en, word_de, caption_en, caption_de, image, frcnn) in enumerate(zip(words_en, words_de, captions_en, captions_de, images, frcnns)):
            d = {'image':image, 'caption_en':caption_en, 'caption_de':caption_de, 'frcnn':frcnn, 'word_en': word_en, 'word_de': word_de}
            json.dump(d, outfile)
            outfile.write('\n')
            

