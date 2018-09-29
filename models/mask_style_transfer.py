import os
import gc
import sys
import random
import math
import numpy as np
import skimage.io
import time

from skimage.color import rgb2gray
from scipy.ndimage.filters import median_filter
from scipy.misc import imread, imsave, toimage

from keras import backend as K
from keras.optimizers import Adam

from models.src.loss import dummy_loss
from models.src.img_util import preprocess_reflect_image, crop_image, check_resize_img
from models.src.nets import image_transform_net

import tensorflow as tf

from io import BytesIO
from models.mask_style.mrcnn import utils
from models.mask_style.mrcnn import visualize
import models.mask_style.mrcnn.model as modellib

from models.mask_style.mask_utils import create_mask
from models.mask_style.mask_utils import test_create_mask
import models.mask_style.coco.coco as coco



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Root directory of the project
ROOT_DIR = os.path.abspath("models/mask_style/")
STYLE_MODEL_DIR = os.path.abspath("models/fast_style_transfer/pretrained/")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained/mask_rcnn_coco.h5")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
config = InferenceConfig()


def split_path(path):
    #drop file extension
    filename = path.rsplit('.', 1)[0]
    #drop static/img/
    filename = filename[11:]
    return filename



def detect_mask(image):
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


def apply_style_mask(content, generated, mask):
    width, height, channels = generated.shape

    for i in range(width):
        for j in range(height):
            if mask[i, j].all() == 0.:
                generated[i, j, :] = content[i, j, :]

    return generated


def apply_style(content, style):
    print(style)
    style= style

    input_file = content

    media_filter = 3

    aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)

    img_width= img_height = x.shape[1]
    model = image_transform_net(img_width,img_height)

    model.compile(Adam(),  dummy_loss)  # Dummy loss since we are learning from regularizes

    model.load_weights(os.path.join(STYLE_MODEL_DIR, style+'_weights.h5'),by_name=True)

    t1 = time.time()
    y = model.predict(x)[0]
    y = crop_image(y, aspect_ratio)

    print("process: %s" % (time.time() -t1))

    ox = crop_image(x[0], aspect_ratio)

    y =  median_filter_all_colours(y, media_filter)

    del model
    K.clear_session()
    gc.collect()
    return y

def transfer(base_image, foresyle_image_path, backstyle_image_path, media_filter=3):
    start_time = time.time()
    base_image = imread(base_image, mode="RGB")
    input_file = check_resize_img(base_image)
    forestyle = split_path(foresyle_image_path)
    backstyle = split_path(backstyle_image_path)

    print(forestyle)
    print(backstyle)
    media_filter = media_filter

    mask_i = detect_mask(input_file)
    
    style_transfer_time = time.time()
    
    generate_1 = apply_style(style=forestyle, content=input_file)

    generate_2 = apply_style(style=backstyle, content=input_file)

    generate_3 = apply_style_mask(content=generate_1, generated=generate_2, mask=mask_i[: , :, 0])
    print("total_transfer_time:", time.time() - start_time)

    output = BytesIO()
    im = toimage(generate_3)
    im.save(output, format='JPEG')
    
    K.clear_session()
    gc.collect()
    return output.getvalue()
