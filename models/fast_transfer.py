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
from io import BytesIO
from models.src.layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from models.src.loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from models.src.img_util import preprocess_reflect_image, crop_image, blend_with_original_colors, original_color_transform, median_filter_all_colours

import models.src.nets as nets
import models.file_path as file_path

def split_path(path):
    """Split image file path

    Params
        ======
            path (str): file path
    """
    #drop file extension
    filename = path.rsplit('.', 1)[0]
    #drop static/img/
    filename = filename[11:]
    return filename
    
def transfer(base_image, syle_image_path, color_adjusting_mode=0, blending_alpha=0, median_filter_size=3):
    """Fast style transformation

    Params
        ======
            base_image (ndarray): content image passed by user
            syle_image_path (str): style image stored in server
            color_adjusting_mode (float): Color adjusting mode. 
                0: None,
                1: Preserve color, 
                2: Blend with original color
            blending_alpha (float): the degree of the blending images, from 0 to 100
            median_filter_size (int): the size of the median filter

    """
    style = split_path(syle_image_path)
    input_file = base_image
    color_adjusting_mode = int(color_adjusting_mode)
    blending_alpha = float(blending_alpha) / 100  # scale to 0 ~ 1
    median_filter_size = median_filter_size
    """ Preprocessing """
    aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)
    img_width = img_height = x.shape[1]
    
    """ Load Model """
    model = nets.depthwise_image_transform_net(img_width, img_height)
    model.compile(Adam(),  dummy_loss) 
    model.load_weights(file_path.MODELS_PATH + "/fast_style_transfer/pretrained/"+style+'_weights.h5')
    print('Model loaded')
    
    """ Start transfer """
    t1 = time.time()
    y = model.predict(x)[0]

    """ Post processing """
    y = crop_image(y, aspect_ratio)
    print("process: %s" % (time.time() -t1))
    ox = crop_image(x[0], aspect_ratio)
    y =  median_filter_all_colours(y, median_filter_size)

    """ Color adjusting """
    if color_adjusting_mode == 1:
        y = original_color_transform(ox, y)
    elif color_adjusting_mode == 2:
        y = blend_with_original_colors(ox, y, blending_alpha)
    
    """ Return the processed image """
    output = BytesIO()
    im = toimage(y)
    im.save(output, format='JPEG')
    del model
    K.clear_session()
    gc.collect()
    return output.getvalue()
