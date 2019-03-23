import os
import gc
import sys
import random
import math
import numpy as np
import skimage.io
import time

from skimage.color import rgb2gray
from scipy.misc import imread, imsave, toimage

from keras import backend as K
from keras.optimizers import Adam

from models.src.loss import dummy_loss
from models.src.img_util import preprocess_reflect_image, crop_image, check_resize_img, blend_with_original_colors, original_color_transform, median_filter_all_colours
from models.src.nets import image_transform_net, depthwise_image_transform_net

import tensorflow as tf

from io import BytesIO
from models.mask_style.mrcnn import utils
from models.mask_style.mrcnn import visualize
import models.mask_style.mrcnn.model as modellib

from models.mask_style.mask_utils import create_mask
from models.mask_style.mask_utils import test_create_mask
import models.mask_style.coco.coco as coco
import models.file_path as file_path



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Root directory of the project
ROOT_DIR = file_path.MODELS_PATH + "./mask_style/"
STYLE_MODEL_DIR = file_path.MODELS_PATH +"./fast_style_transfer/pretrained/"
# Directory to save logs and trained model
MODEL_DIR = ROOT_DIR + "logs"
# Local path to trained weights file
COCO_MODEL_PATH = ROOT_DIR + "pretrained/mask_rcnn_coco.h5"
# Import Mask RCNN
config = InferenceConfig()


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



def detect_mask(image):
    """Using Mask-RCNN to detect object and generate masks

    Params
        ======
            image (ndarray): image

    """
    mrcnn_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    mrcnn_model.load_weights(COCO_MODEL_PATH, by_name=True)
    results = mrcnn_model.detect([image], verbose=0)    
    r = results[0]
    mask_i = create_mask(image, r['rois'], r['masks'], r['class_ids'], r['scores'])
    del mrcnn_model
    K.clear_session()
    gc.collect()
    
    return mask_i

def apply_style_mask(content, generated, mask):
    """Apply masks on content image

    Params
        ======
            content (ndarray): content image
            generated (ndarray): generated image
            mask (ndarray): mask
    """
    width, height, channels = generated.shape

    for i in range(width):
        for j in range(height):
            if mask[i, j].all() == 0.:
                generated[i, j, :] = content[i, j, :]

    return generated


def apply_style(content, style, median_filter_size=3):
    """Apply style using fast style transformation

    Params
        ======
            content (ndarray): content image passed by user
            style (str): style image stored in server

    """
    style= style
    input_file = content
    median_filter_size = median_filter_size

    """ Preprocessing """
    aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)
    img_width= img_height = x.shape[1]

    """ Load Model """
    model = depthwise_image_transform_net(img_width, img_height)
    model.compile(Adam(),  dummy_loss)  # Dummy loss since we are learning from regularizes
    model.load_weights(STYLE_MODEL_DIR + style + '_weights.h5')

    """ Start transfer """
    t1 = time.time()
    y = model.predict(x)[0]
    print("process: %s" % (time.time() -t1))

    """ Post processing """
    y = crop_image(y, aspect_ratio)
    y =  median_filter_all_colours(y, median_filter_size)

    del model
    K.clear_session()
    gc.collect()
    return y

def transfer(content_image, foresyle_image_path, backstyle_image_path, color_adjusting_mode=0, blending_alpha=0, median_filter_size=3):
    """Mask style transformation

    Params
        ======
            content_image (ndarray): content image passed by user
            foresyle_image_path (str): foreground style image stored in server
            backstyle_image_path (str): background style image stored in server
            color_adjusting_mode (float): Color adjusting mode. 
                0: None,
                1: Preserve color, 
                2: Blend with original color
            blending_alpha (float): the degree of the blending images, from 0 to 100
            median_filter_size (int): the size of the median filter

    """
    content_image = imread(content_image, mode="RGB")
    color_adjusting_mode = int(color_adjusting_mode)
    blending_alpha = float(blending_alpha) / 100  # scale to 0 ~ 1
    median_filter_size = median_filter_size

    """ Preprocessing """
    content_image = check_resize_img(content_image)
    aspect_ratio, x = preprocess_reflect_image(content_image, size_multiple=4)
    img_width = img_height = x.shape[1]

    forestyle = split_path(foresyle_image_path)
    backstyle = split_path(backstyle_image_path)

    print(forestyle)
    print(backstyle)
    """ Start transfer """
    start_time = time.time()
    mask_i = detect_mask(content_image)
    generated_1 = apply_style(style=forestyle, content=content_image)
    print("generated_1", generated_1.shape)
    generated_2 = apply_style(style=backstyle, content=content_image)
    print("generated_2", generated_2.shape)

    generated_3 = apply_style_mask(content=generated_1, generated=generated_2, mask=mask_i[: , :, 0])
    print("generated_3", generated_3.shape)
    print("total_transfer_time:", time.time() - start_time)

    ox = crop_image(x[0], aspect_ratio)
    """ Color adjusting """
    print(color_adjusting_mode)
    if color_adjusting_mode == 1:
        y = original_color_transform(ox, generated_3)
    elif color_adjusting_mode == 2:
        y = blend_with_original_colors(ox, generated_3, blending_alpha)
    else:
        y = generated_3

    output = BytesIO()
    im = toimage(y)
    im.save(output, format='JPEG')
    
    K.clear_session()
    gc.collect()
    return output.getvalue()
