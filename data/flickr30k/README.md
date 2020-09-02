0. create a python 2.7 conda environment:

   conda create -n cc python=2.7 pip
   conda activate cc
   pip install Cython numpy Pillow

1. download from https://github.com/multi30k/dataset the Multi30k dataset and place all images under the 
   directory flickr30k/flickr30k-images and train.de, train.en, val.en, val.de, test_2016.en, test_2016.de
   under flickr30k/
   
2. cd to /flickr30k

3. sh split_flickr.sh

4. cd to /utils/

5. python gen_train_image_json.py
   python gen_val_image_json.py
   python gen_test_image_json.py


6. 1) zip (without compression) "train_image" by
   
   cd ../train_image
   zip -0 ../train_image.zip ./*
   cd ../utils/
   
   2) zip (without compression) "val_image" by
   
   cd ../val_image
   zip -0 ../val_image.zip ./*
   cd ../utils/

   3) zip (without compression) "test_image" by
   
   cd ../test_image
   zip -0 ../test_image.zip ./*
   cd ../utils/   
   
   
   
7. git clone https://github.com/jackroos/bottom-up-attention and follow "Installation" :

   1) Build the Cython modules

   cd $REPO_ROOT/lib
   make
   
   2) Build Caffe and pycaffe

   cd $REPO_ROOT/caffe
   # Now follow the Caffe installation instructions here:
   #   http://caffe.berkeleyvision.org/installation.html

   # If you're experienced with Caffe and have all of the requirements installed
   # and your Makefile.config in place, then simply do:
   make -j8 && make pycaffe
   
   3) Download pretrained model (https://www.dropbox.com/s/wqada4qiv1dz9dk/resnet101_faster_rcnn_final.caffemodel?dl=1), and put it under data/faster_rcnn_models.
   
9. python ./tools/generate_tsv_v2.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split flickr30k_train --data_root {Flickr_30k_Root} --out {Flickr_30k_Root}/train_frcnn/

python ./tools/generate_tsv_v2.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split flickr30k_val --data_root {Flickr_30k_Root} --out {Flickr_30k_Root}/valfrcnn/

python ./tools/generate_tsv_v2.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split flickr30k_test --data_root {Flickr_30k_Root} --out {Flickr_30k_Root}/test_frcnn/


10. zip (without compression) "train_frcnn", "val_frcnn" and "test_frcnn" similar to step 6.
