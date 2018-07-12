Tucana Neural Style
===============================

This website can transfer an image into a stylish image by chosen a pretrained style.

The fast neural style part is base on [fast-neural-style-keras](https://github.com/misgod/fast-neural-style-keras) by misgod, We made some changes to it.

And the original neural style paper is from [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.

the fast neural style is from [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson et al.

Environment
===============================

*  Python 3.5 (Recommend to use Anaconda)

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

You have to download pretrained models or train a model first.

Go to model folder and put an image into images/style 

and go back to model type the command, for example

```
python train.py -s la_muse -o pretrained/la_muse_model
```


use console to the folder and type

```
python app.py
```

and open browser go to **localhost:5000**