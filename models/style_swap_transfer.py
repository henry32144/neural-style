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
from models.src.loss import dummy_loss
from models.src.img_util import style_swap_preprocess_image, crop_image

from models.src.nets import InverseNet_3_3_res, build_encode_net

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


def transfer(base_image, syle_image, original_color=0, blend_alpha=0, media_filter=3):
    input_file = base_image
    style_file = syle_image
    
    original_color = original_color
    blend_alpha = blend_alpha
    media_filter = media_filter
    aspect_ratio = 1
    
    content_processed = style_swap_preprocess_image(input_file)
    style_processed = style_swap_preprocess_image(style_file)

    img_width = img_height = content_processed.shape[1]
    print(img_width, img_height)
    
    encode_net = build_encode_net((img_width, img_height, 3))
    print('encode_net')
    
    inverse_net = InverseNet_3_3_res((int(img_width/4) ,int(img_height/4) ,256))
    inverse_net.load_weights("./models/style_swap/pretrained/inverse_net_13_res_nor.h5", by_name=True)
    print('Model loaded')
    
    inverse_net.compile(optimizer="adam", loss='mse')
    inverse_net.summary()
    image_feature = encode_net.predict([content_processed, style_processed])


    t1 = time.time()
    y = inverse_net.predict([image_feature])[0]
    y = crop_image(y, aspect_ratio)

    print("process: %s" % (time.time() -t1))

    ox = crop_image(content_processed[0], aspect_ratio)

    #y =  median_filter_all_colours(y, media_filter)

    if blend_alpha > 0:
        y = blend(ox,y,blend_alpha)


    if original_color > 0:
        y = original_colors(ox,y,original_color )
        
    output = BytesIO()
    im = toimage(y)
    im.save(output, format='JPEG')
    
    del encode_net
    del inverse_net
    K.clear_session()
    gc.collect()
    return output.getvalue()
