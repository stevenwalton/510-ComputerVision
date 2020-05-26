# Assignment 2

Submit:
- a single PDF report
- a single Colab/Jupyter notebook  (.ipynb) for code with comments.
- a single zip file if need

Filename: HW2_YourName.(pdf/ipynb/zip)

Due: Sunday 05/17/2020 11:55PM.

Dataset:

[PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)

Part 1: (10/15 pts)

Implement a very simple object detector using the knowledge you have learned so
far.

For example, you can use selective search to generate proposals from images,
and then crop the regions from input image, classify those with pre-trained
image classifiers, and perform NMS.

Feel free to find open sources implementations for components (such as
selective search, NMS etc.)

Report on validation set:
- recall of region proposal generation
- mAP of object detection
- visualization of object detection results on some of the images

Part 2: (5/15 pts)

Find the Faster RCNN models from
[mmdetection](https://github.com/open-mmlab/mmdetection) (or other popular
detection codebases), and test some of them (at least two):

Report on validation set:
- recall of region proposal generation
- mAP of object detection
- visualization of object detection results on some of the images
 and compare the results with those from part-1

 

Extra Credits: (5 pts, pick either one)

(a) find a way to train (on VOC train set) the simple detector you wrote and
compare the results (on VOC val set) with those from part-1

(b)  fine-tune a pre-trained detector to recognize new object classes (you can
ref to this
[tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html))

## Retraining
40 epochs lr=0.0001 mAP -> 0.00397
Take2 mAP -> 0.0041 with 0 except aeroplane. Recall: 0.1 AP: 0.082
100 epochs mAP -> 0.00362: All zero except aeroplane. Recall: 0.0819 AP 0.0724
VGG16 20 epochs mAP -> 0.0112: all zero except airplane. Recall: 0.246, AP
0.2245
VGG11 20 epochs mAP -> 0.0107: Same Recall 0.2341 AP 0.2137
VGG19 mAP -> 0.1152: Recall 0.2463 AP 0.2304
