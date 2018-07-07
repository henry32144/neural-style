Tucana Neural Style
===============================

This website can transfer an image into a stylish image by chosen a pretrained style.

The fast neural style part is base on [fast-neural-style-keras](https://github.com/misgod/fast-neural-style-keras) by misgod.

and the original neural style paper is from [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.

the fast neural style is from [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson et al.

Environment
===============================

*  Python 3.5 (Recommend use Anaconda)

*  CUDA 8.0
*  Cudnn 6.0

*  Flask 0.12.2
*  Keras 2.1.3
*  tensorflow-gpu 1.4
*  scipy 1.0.0

or directly use pip install the requirement txt file

```
pip -r requirement.txt
```

How to Use
==============================

use console to the folder and type

```
python app.py
```

and open browser go to **localhost:5000**