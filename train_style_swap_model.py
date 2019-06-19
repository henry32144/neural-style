from keras.layers import Input, merge
from keras.models import Model,Sequential
from keras.optimizers import Adam, SGD,Nadam,Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
from keras import backend as K
from scipy.misc import imsave, toimage

import os
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

from models.src.nets import InverseNet_3_1_with_encoder, build_encode_net_with_swap_3_1

IMG_HEIGHT = 256
IMG_WIDTH = 256
tv_weight = 1e-6



def show_without_deprocess(x, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, save=False, name='', iterate=0):

    x = x.reshape((IMG_WIDTH, IMG_HEIGHT, 3))
    img = np.clip(x, 0, 255).astype('uint8')
    
    aspect_ratio = get_ratio(img)  
    img_ht = int(IMG_WIDTH * aspect_ratio)
    #print("Rescaling Image to (%d, %d)" % (IMG_WIDTH, img_ht))
    img = imresize(img, (IMG_WIDTH, img_ht), interp='bilinear')
    im = toimage(img)
    if save:
        dir_name = "output"
        filename = dir_name + "/%s_%d.jpg" % (name, iterate)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        imsave(filename, im)
    else:
        plt.imshow(im)

def main(args):
    tv_weight= args.tv_weight
    img_width = img_height =  args.image_size
    style_image_path = args.style
    content_image_path = args.path

    # VGG feature extract
    encode_net = build_encode_net_with_swap_3_1((img_height, img_width, 3))
    print('Encode Model loaded')
    inverse_net = InverseNet_3_1_with_encoder((int(img_height / 4) , int(img_width / 4), 256), tv_weight)
    inverse_net.load_weights(file_path.KERAS_MODELS_PATH + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
    print('Model loaded')
    inverse_net.compile(optimizer="adam", loss='mse')

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    epoch = 80000
    batch_size = 1

    content_gen = datagen.flow_from_directory(content_image_path,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=batch_size,
                                                class_mode=None)
        
    style_gen = datagen.flow_from_directory(style_image_path,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=batch_size,
                                                class_mode=None)

    # Used for output image
    output_image_func = K.function([inverse_net.input], [inverse_net.layers[-9].output])
   
    total_loss = []
    start_time = time.time()
    for i in range(epoch):
        try:
            content = content_gen.next()
            style = style_gen.next()
        except:
            print("error image: ", style_gen.filenames[-1])
        
        swapped_feature = encode_net.predict([content,style])
        history = inverse_net.train_on_batch([swapped_feature], swapped_feature)
        
        if i % 10 == 0:
            
            total_loss.append(history)
        
        if i % 100 == 0:
            print("epoch: %d" % (i))
            print("mse loss: %f" % history)

    end_time = time.time()
    print("cost time: %d min" % int((end_time - start_time) / 60))
    
    inverse_net.save_weights("models/style_swap/pretrained/inverse_net_vgg19.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a style swap model')
    
    parser.add_argument('--path', '-p', type=str, required=True,
                        help='training images path')
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style images path')
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file path without extension')
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--image_size', default=256, type=int)

    args = parser.parse_args()
    main(args)