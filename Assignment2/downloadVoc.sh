#!/bin/bash

rm -r VOC*
mkdir VOCTest VOCTrainVal
echo "Downloading and untaring Test"
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar -C VOCTest
echo "Doanloading and untaring Trainval"
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar -C VOCTrainVal/

