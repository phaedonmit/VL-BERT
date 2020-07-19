import jsonlines

filepath = "/experiments/faidon/VL-BERT/data/flickr30k/train_MLT_frcnn.json"
database = list(jsonlines.open(filepath))
ambiguous_german = []

for entry in database:
    if entry['word_de'] not in ambiguous_german:
        ambiguous_german.append(entry['word_de'])
        print(entry['word_de'])


with open('MLT_vocab.txt', 'w') as f:
    for word in ambiguous_german:
        f.write("%s\n" % word)