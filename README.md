Tucana Neural Style in Keras
===============================

You can use this website to transfer an image into a stylish image by three methods.

this project is implemented mainly in Keras(some part is Tensorflow)

Reference
===============================

**Fast Neural Style**

The fast neural style implementation is base on [fast-neural-style-keras](https://github.com/misgod/fast-neural-style-keras) by misgod, We made some changes to it,

and this method is base on [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson et al.

**Style Swap**

The style swap layer is reference from [WCT-TF](https://github.com/eridgd/WCT-TF) by eridgd, 

the original paper is [Fast Patch-based Style Transfer of Arbitrary Style](https://arxiv.org/abs/1612.04337) by Chen et al.

**Mask Style**

We use [Mask R-CNN](https://github.com/matterport/Mask_RCNN) which is implemented by [matterport](https://github.com/matterport),

the original paper of Mask R-CNN is [Mask R-CNN](https://arxiv.org/abs/1703.06870) by He et al.

Our Environment
===============================

*  Python 3.5 (Anaconda)

*  CUDA 8.0
*  Cudnn 6.0

*  Flask 0.12.2
*  Keras 2.1.3
*  tensorflow-gpu 1.4
*  scipy 1.0.0
*  scikit-image
*  pycocotool
*  imgaug
*  cython

or directly use pip install the requirement txt file

```
pip install -r requirement.txt
```

How to Use
==============================

Download the [pretrained models](https://drive.google.com/open?id=1mbi7981rvRvqHf6blBhVcvQMV5yCa_i2)

and put the models folder into the project root folder and overwrite it,

then modify MODELS_PATH in "models/file_path.py",

use console to the folder and type

```
python app.py
```

and open browser go to **localhost:5000**
