import os
import jsonlines

filenames = [
        "DEIMG",
        "ENIMG",
        "TUIMG",
        "DEENIMG",
        "ENDEIMG",
        "ENFRIMG",
        "FRENIMG",
        "DEEN",
        "ENDE",
        "ENFR",
        "ENTU",
        "FREN",
        "TUEN"
]

source_files = [
        "/data/ozan/datasets/mmbert/flickr30k_de/test_frcnn.json",
        "/data/ozan/datasets/mmbert/flickr30k_en/test_frcnn.json",
        "/data/ozan/datasets/mmbert/flickr8k_tr/test_frcnn.json",
        "/data/ozan/datasets/mmbert/multi30k_en_de/test_frcnn.json",
        "/data/ozan/datasets/mmbert/multi30k_en_de/test_frcnn.json",
        "/data/ozan/datasets/mmbert/multi30k_en_fr/test_frcnn.json",
        "/data/ozan/datasets/mmbert/multi30k_en_fr/test_frcnn.json",
        "/data/ozan/datasets/mmbert/iwslt14_en_de/test_frcnn.json",
        "/data/ozan/datasets/mmbert/iwslt14_en_de/test_frcnn.json",
        "/data/ozan/datasets/mmbert/iwslt14_en_fr/test_frcnn.json",
        "/data/ozan/datasets/mmbert/setimes_en_tr/test_frcnn.json",
        "/data/ozan/datasets/mmbert/iwslt14_en_fr/test_frcnn.json",
        "/data/ozan/datasets/mmbert/setimes_en_tr/test_frcnn.json"
]

captions = [
        "caption_de",
        "caption_en",
        "caption_de",
        "caption_en",
        "caption_de",
        "caption_de",
        "caption_en",
        "caption_en",
        "caption_de",
        "caption_de",
        "caption_de",
        "caption_en",
        "caption_en"
]

base_dir = '/data/faidon/VL-BERT/data/ground_truth'

for filename, source_file, caption in zip(filenames,source_files, captions):
    result_path = os.path.join(base_dir, '{}.txt'.format(filename))
    ground_truths = []
    

    database = list(jsonlines.open(source_file))
    if filename=="TUIMG":
        for start in range(2):
            result_path = os.path.join(base_dir, '{}_{}.txt'.format(filename, str(start)))
            for i in range(start, len(database), 2)
                with open(result_path, 'w') as f:
                    for line in database:
                        f.write('%s\n' % line[caption])

    elif filename=="DEIMG" or filename=="ENIMG":
        for start in range(5):
            result_path = os.path.join(base_dir, '{}_{}.txt'.format(filename, str(start)))
            for i in range(start, len(database), 5)
                with open(result_path, 'w') as f:
                    for line in database:
                        f.write('%s\n' % line[caption])    
    else:
        with open(result_path, 'w') as f:
            for line in database:
                f.write('%s\n' % line[caption])