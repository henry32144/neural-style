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
import models.file_path as file_path

from skimage import color, exposure, transform
from scipy import ndimage
from io import BytesIO
from models.src.loss import dummy_loss
from models.src.img_util import style_swap_preprocess_image, crop_image, blend_with_original_colors, original_color_transform, median_filter_all_colours

from models.src.nets import InverseNet_3_1, build_encode_net_with_swap_3_1


def transfer(base_image, style_image, color_adjusting_mode=0, blending_alpha=0, median_filter_size=3):
    """Style swap transformation

    Params
        ======
            base_image (ndarray): content image passed by user
            style_image (ndarray): style image passed by user
            color_adjusting_mode (float): Color adjusting mode. 
                0: None,
                1: Preserve color, 
                2: Blend with original color
            blending_alpha (float): the degree of the blending images, from 0 to 100
            median_filter_size (int): the size of the median filter

    """
    input_file = base_image
    style_file = style_image
    IMG_WIDTH = IMG_HEIGHT = 700

    color_adjusting_mode = int(color_adjusting_mode)
    blending_alpha = float(blending_alpha) / 100  # scale to 0 ~ 1
    median_filter_size = median_filter_size

    aspect_ratio = 1
    
    """ Preprocessing """
    content_original = style_swap_preprocess_image(input_file, IMG_WIDTH, IMG_HEIGHT, preserve_original=True)
    content_processed = style_swap_preprocess_image(input_file, IMG_WIDTH, IMG_HEIGHT)
    style_processed = style_swap_preprocess_image(style_file, IMG_WIDTH, IMG_HEIGHT)

    img_width = img_height = content_processed.shape[1]
    print(img_width, img_height)
    
    """ Load Model """
    encode_net = build_encode_net_with_swap_3_1((img_width, img_height, 3))
    print('Encode Model loaded')
    inverse_net = InverseNet_3_1((int(img_width / 4) , int(img_height / 4), 256))
    inverse_net.load_weights(file_path.MODELS_PATH + "/style_swap/pretrained/inverse_net_vgg19.h5", by_name=True)
    print('Model loaded')
    inverse_net.compile(optimizer="adam", loss='mse')

    """ Start transfer """
    t1 = time.time()
    image_feature = encode_net.predict([content_processed, style_processed])
    y = inverse_net.predict([image_feature])[0]
    print("process: %s" % (time.time() -t1))

    """ Post processing """
    y = crop_image(y, aspect_ratio)
    y =  median_filter_all_colours(y, median_filter_size)
    ox = crop_image(content_original[0], aspect_ratio)
    
    """ Color adjusting """
    if color_adjusting_mode == 1:
        y = original_color_transform(ox, y)
    elif color_adjusting_mode == 2:
        y = blend_with_original_colors(ox, y, blending_alpha)
    
    y = y[10:-10, 10:-10]
    """ Return the processed image """
    output = BytesIO()
    im = toimage(y)
    im.save(output, format='JPEG')
    del encode_net
    del inverse_net
    K.clear_session()
    gc.collect()
    return output.getvalue()
