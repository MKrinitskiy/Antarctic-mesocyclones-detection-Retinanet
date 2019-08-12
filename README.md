## [Keras RetinaNet](https://github.com/fizyr/keras-retinanet) adaptation for antarctic mesocyclones detection

Keras implementation of RetinaNet object detection forked from the original one (see link above) described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This version was modified in order to fit the requirements of the problem:

- source data is stored in numpy arrays
- source data along with the labels were augmented in a particular way (see preprocessing repository  [here](https://github.com/MKrinitskiy/Antarctic-MCs-detection-Train-data-collect))
- satellite mosaics are too large, so cropping was involved at the [preprocessing](https://github.com/MKrinitskiy/Antarctic-MCs-detection-Train-data-collect) stage