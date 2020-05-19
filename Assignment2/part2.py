import os
from os.path import exists, join, basename, splitext
import sys
sys.path.append('mmdetection/')
import matplotlib
import matplotlib.pylab as plt
import glob
import pandas as pd
import xml.etree.ElementTree as ET

MODELS_CONFIG = {
    'faster_rcnn_r50_fpn_1x': {
        'config_file': 'mmdetection/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py'
    },
    'cascade_rcnn_r50_fpn_1x': {
        'config_file': 'mmdetection/configs/cascade_rcnn_r50_fpn_1x.py',
    },
    'retinanet_r50_fpn_1x': {
        'config_file': 'mmdetection/configs/retinanet_r50_fpn_1x.py',
    }
}


use = 'faster_rcnn_r50_fpn_1x'
epochs = 10
config_file = MODELS_CONFIG[use]['config_file']

plt.rcParams['axes.grid'] = False

class_names = []
xml_list = []
