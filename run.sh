# wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# unzip VOCtrainval_11-May-2012.tar

python train.py --save "./results/pascal/vgg_1shot/1shots" -weight 1.0
