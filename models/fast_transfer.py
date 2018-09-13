from keras.layers import Input, merge
from keras.models import Model,Sequential
from keras.optimizers import Adam, SGD,Nadam,Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imsave, toimage
import time
import numpy as np 
import argparse
import h5py
import tensorflow as tf
import gc

from skimage import color, exposure, transform
from scipy import ndimage
from scipy.ndimage.filters import median_filter
from io import BytesIO
from models.src.layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from models.src.loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from models.src.img_util import preprocess_reflect_image, crop_image

import models.src.nets as nets


# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized,original_color):
    # Histogram normalization in v channel
    ratio=1. - original_color 

    hsv = color.rgb2hsv(original/255)
    hsv_s = color.rgb2hsv(stylized/255)

    hsv_s[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
    img = color.hsv2rgb(hsv_s)    
    return img

def blend(original, stylized, alpha):
    return alpha * original + (1 - alpha) * stylized



def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:,:,d], size=(window_size,window_size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    
    return im_conv

def load_weights(model,file_path):
    f = h5py.File(file_path)

    layer_names = [name for name in f.attrs['layer_names']]

    for i, layer in enumerate(model.layers[:31]):
        g = f[layer_names[i]]
        weights = [g[name] for name in g.attrs['weight_names']]
        layer.set_weigh
        ts(weights)

    f.close()
    
    print('Pretrained Model weights loaded.')

def split_path(path):
    #drop file extension
    filename = path.rsplit('.', 1)[0]
    #drop static/img/
    filename = filename[11:]
    return filename
    
def transfer(base_image, syle_image_path, original_color=0, blend=0, media_filter=3):
    style = split_path(syle_image_path)
    input_file = base_image
    original_color = original_color
    blend_alpha = blend
    media_filter = media_filter

    #aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)
    
    aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)

    #img_width = x.shape[1]
    #img_height = x.shape[2]
    img_width = img_height = x.shape[1]
    net = nets.image_transform_net(img_width,img_height)
    model = nets.loss_net(net.output,net.input,img_width,img_height,"",0,0)

    model.compile(Adam(),  dummy_loss)  # Dummy loss since we are learning from regularizes

    model.load_weights("./models/fast_style_transfer/pretrained/"+style+'_weights.h5',by_name=False)
    print('Model loaded')
    
    t1 = time.time()
    y = net.predict(x)[0] 
    y = crop_image(y, aspect_ratio)

    print("process: %s" % (time.time() -t1))

    ox = crop_image(x[0], aspect_ratio)

    y =  median_filter_all_colours(y, media_filter)

    if blend_alpha > 0:
        y = blend(ox,y,blend_alpha)


    if original_color > 0:
        y = original_colors(ox,y,original_color )
        
    output = BytesIO()
    im = toimage(y)
    im.save(output, format='JPEG')
    
    del model
    del net
    K.clear_session()
    gc.collect()
    return output.getvalue()
