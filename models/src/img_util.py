from scipy.misc import imread, imresize, imsave, fromimage, toimage
from skimage import color
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from PIL import Image
import numpy as np
import math
import os
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from scipy.ndimage.filters import median_filter
from keras.applications.vgg19 import preprocess_input

def median_filter_all_colours(im_small, window_size):
    """Applies a median filer to all colour channels
    
    Params
        ======
            im_small (ndarray): image
            window_size (int): the size of the filter
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:,:,d], size=(window_size, window_size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    
    return im_conv

# Util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, img_width=256, img_height=256, load_dims=False, resize=True, size_multiple=4):
    '''
    Preprocess the image so that it can be used by Keras.
    Args:
        image_path: path to the image
        img_width: image width after resizing. Optional: defaults to 256
        img_height: image height after resizing. Optional: defaults to 256
        load_dims: decides if original dimensions of image should be saved,
                   Optional: defaults to False
        vgg_normalize: decides if vgg normalization should be applied to image.
                       Optional: defaults to False
        resize: whether the image should be resided to new size. Optional: defaults to True
        size_multiple: Deconvolution network needs precise input size so as to
                       divide by 4 ("shallow" model) or 8 ("deep" model).
    Returns: an image of shape (3, img_width, img_height) for dim_ordering = "th",
             else an image of shape (img_width, img_height, 3) for dim ordering = "tf"
    '''
    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        global img_WIDTH, img_HEIGHT, aspect_ratio
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    if resize:
        if img_width < 0 or img_height < 0: # We have already loaded image dims
            img_width = (img_WIDTH // size_multiple) * size_multiple # Make sure width is a multiple of 4
            img_height = (img_HEIGHT // size_multiple) * size_multiple # Make sure width is a multiple of 4
        img = imresize(img, (img_width, img_height),interp='nearest')

    if K.image_dim_ordering() == "th":
        img = img.transpose((2, 0, 1)).astype(np.float32)
    else:

        img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image_for_generating(image_path, size_multiple=4):
    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    org_w = img.shape[0]
    org_h = img.shape[1]

    aspect_ratio = org_h/org_w

    size  = org_w if org_w > org_h else org_h

    pad_w = (size - org_w) // 2
    pad_h = (size - org_h) // 2

    tf_session = K.get_session()
    kvar = K.variable(value=img)

    paddings = [[pad_w,pad_w],[pad_h,pad_h],[0,0]]
    squared_img = tf.pad(kvar,paddings, mode='REFLECT', name=None)
    img = K.eval(squared_img)

    
    img_width = (squared_img.shape[1] // size_multiple) * size_multiple # Make sure width is a multiple of 4
    img_height = (squared_img.shape[0] // size_multiple) * size_multiple # Make sure width is a multiple of 4

    img = imresize(img, (img_width, img_height),interp='nearest')

    if K.image_dim_ordering() == "Th":
        img = img.transpose((2, 0, 1)).astype(np.float32)
    else:

        img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)
    return ((org_w,org_h), (img_width.value,img_height.value),img)


def check_resize_img(im):
    width = im.shape[0]
    height = im.shape[1]
    #1024 * 768 * 3
    threshold = 2359296
    if im.size >= threshold:
        if width >= height:
            new_width = int(math.sqrt(threshold/1.25))
            new_height = int(new_width * height * 1.0 / width)
        else:
            new_height = int(math.sqrt(threshold/1.25))
            new_width = int(new_height * width * 1.0 / height)
        im = imresize(im, (new_width, new_height), interp='bilinear')
    return im

def style_swap_preprocess_image(image_path, IMG_WIDTH=512, IMG_HEIGHT=512, preserve_original=False):
    mode = "RGB"
    img = imread(image_path, mode=mode)

    img = imresize(img,(IMG_WIDTH, IMG_HEIGHT), interp='bilinear').astype('float32')

    if not preserve_original:
        img = preprocess_input(img)
    
    if K.image_dim_ordering() == "th":
        img = img.transpose((2, 0, 1)).astype('float32')

    img = np.expand_dims(img, axis=0)
    print(img.shape)
    return img


def preprocess_reflect_image(image_path, size_multiple=4):
    if(type(image_path) is not np.ndarray):    
        img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    else:
        img = image_path
    img = check_resize_img(img) # check image size
    org_w = img.shape[0]
    org_h = img.shape[1]
    aspect_ratio = org_h/org_w
    
    print(org_w,',', org_h,',', aspect_ratio)

    
    sw = (org_w // size_multiple) * size_multiple # Make sure width is a multiple of 4
    sh = (org_h // size_multiple) * size_multiple # Make sure width is a multiple of 4

    
    size  = sw if sw > sh else sh

    pad_w = (size - sw) // 2
    pad_h = (size - sh) // 2

    tf_session = K.get_session()
    kvar = K.variable(value=img)

    paddings = [[pad_w,pad_w],[pad_h,pad_h],[0,0]]
    squared_img = tf.pad(kvar,paddings, mode='REFLECT', name=None)
    img = K.eval(squared_img)

    
    img = imresize(img, (size, size),interp='bilinear')
    img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)
    return (aspect_ratio  ,img)


def crop_image(img, aspect_ratio):
    if aspect_ratio >1:
        w = img.shape[0]
        h = int(w / aspect_ratio)
        print('w: ', w, ',h: ',h, 'ratio>1')
        img =  K.eval(tf.image.crop_to_bounding_box(img, (w-h)//2,0,h,w))
    else:
        h = img.shape[1]
        w = int(h * aspect_ratio)
        print('w: ', w, ',h: ',h, 'ratio<1')
        img = K.eval(tf.image.crop_to_bounding_box(img, 0,(h-w)//2,h,w))
    return img


def deprocess_image(x, img_width=256, img_height=256):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# fork from 6o6o. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def blend_with_original_colors(original, stylized, original_color):
    # Histogram normalization in v channel
    ratio=1. - original_color 

    hsv = color.rgb2hsv(original/255)
    hsv_s = color.rgb2hsv(stylized/255)

    hsv_s[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
    img = color.hsv2rgb(hsv_s)    
    return img

# fork from titu1994. https://github.com/titu1994/Neural-Style-Transfer/blob/master/INetwork.py
# util function to preserve image color
def original_color_transform(content, generated, mask=None):
    content = fromimage(toimage(content, mode='RGB'), mode='YCbCr')
    generated = fromimage(toimage(generated, mode='RGB'), mode='YCbCr')  # Convert to YCbCr color space

    if mask is None:
        generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    generated[i, j, 1:] = content[i, j, 1:]

    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated


def load_mask(mask_path, shape, return_mask_img=False):
    if K.image_dim_ordering() == "th":
        _, channels, width, height = shape
    else:
        _, width, height, channels = shape

    mask = imread(mask_path, mode="L") # Grayscale mask load
    mask = imresize(mask, (width, height)).astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    if return_mask_img: return mask

    mask_shape = shape[1:]

    mask_tensor = np.empty(mask_shape)

    for i in range(channels):
        if K.image_dim_ordering() == "th":
            mask_tensor[i, :, :] = mask
        else:
            mask_tensor[:, :, i] = mask

    return mask_tensor