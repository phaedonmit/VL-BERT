# split dataset into validation
mkdir -p /experiments/faidon/VL-BERT/data/flickr30k/val_image/
while read p; do
  cp "/experiments/faidon/VL-BERT/data/flickr30k/flickr30k-images/$p" "/experiments/faidon/VL-BERT/data/flickr30k/val_image/$p"
done </experiments/faidon/VL-BERT/data/flickr30k/utils/val.txt
# and training dataset
mkdir -p /experiments/faidon/VL-BERT/data/flickr30k/train_image/
while read p; do
  cp "/experiments/faidon/VL-BERT/data/flickr30k/flickr30k-images/$p" "/experiments/faidon/VL-BERT/data/flickr30k/train_image/$p"
done </experiments/faidon/VL-BERT/data/flickr30k/utils/train.txt
# and test dataset
mkdir -p /experiments/faidon/VL-BERT/data/flickr30k/test_image/
while read p; do
  cp "/experiments/faidon/VL-BERT/data/flickr30k/flickr30k-images/$p" "/experiments/faidon/VL-BERT/data/flickr30k/test_image/$p"
done </experiments/faidon/VL-BERT/data/flickr30k/utils/test_2016_flickr.txt