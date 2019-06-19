from keras.layers import Input, merge
from keras.models import Model,Sequential
from keras.optimizers import Adam, SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imsave

import time
import numpy as np
import os
import argparse

from keras.callbacks import TensorBoard
from scipy import ndimage
from PIL import Image
from scipy.misc import imread, imresize, imsave, fromimage, toimage

from models.src.layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from models.src.loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
import models.src.nets as nets

def get_ratio(image):
    img = Image.fromarray(image).convert('RGB')
    img_WIDTH, img_HEIGHT = img.size
    aspect_ratio = float(img_HEIGHT) / img_WIDTH
    return aspect_ratio

def show_without_deprocess(x, IMG_WIDTH, IMG_HEIGHT, save=True, name='', iterate=0, style="udnie"):
    x = x.reshape((IMG_WIDTH, IMG_HEIGHT, 3))
    img = np.clip(x, 0, 255).astype('uint8')
    
    # get the ratio of original image
    aspect_ratio = get_ratio(img)  
    img_ht = int(IMG_WIDTH * aspect_ratio)
    #print("Rescaling Image to (%d, %d)" % (IMG_WIDTH, img_ht))
    
    img = imresize(img, (IMG_WIDTH, img_ht), interp='bilinear')
    im = toimage(img)
    if save:
        dir_name = "trained/{}".format(style)
        filename = dir_name + "/{}.jpg".format(iterate)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        imsave(filename, im)
    else:
        plt.imshow(im)

def get_style_img_path(style):
    return "static/img/styles/"+style+".jpg"


def main(args):
    style_weight= args.style_weight
    content_weight= args.content_weight
    tv_weight= args.tv_weight
    style_name = args.style
    img_width = img_height =  args.image_size

    generated_original_style_path="output/{}".format(style_name)
    print(generated_original_style_path)
    style_image_path = get_style_img_path(style_name)
    train_image_path = args.path

    net = nets.depthwise_image_transform_net(img_width,img_height,tv_weight)
    model = nets.multi_loss_net(net.output,net.input,img_width,img_height,style_image_path,content_weight,style_weight)
    model.summary()

    epoch = 82785
    train_batchsize =  1

    learning_rate = 1e-3 #1e-3
    optimizer = Adam()

    datagen = ImageDataGenerator()

    content_gen = datagen.flow_from_directory(train_image_path,
                                          shuffle=False,
                                            target_size=(img_width, img_height),
                                            batch_size=train_batchsize,
                                            class_mode=None)

    style_gen = datagen.flow_from_directory(generated_original_style_path,
                                            shuffle=False,
                                                target_size=(img_width, img_height),
                                                batch_size=train_batchsize,
                                                class_mode=None)

    model.compile(optimizer=optimizer, loss={"transform_output": "mse", "block5_pool": dummy_loss},
               loss_weights={'transform_output': 0.01,
                              'block5_pool': 0.0})
    
    dummy_y = np.zeros((train_batchsize,img_width,img_height,3)) # Dummy output, not used since we use regularizers to train

    total_loss = []
    start_time = time.time()
    print("total epoch: %d \n" % int(epoch/train_batchsize))
    for i in range(int(epoch/train_batchsize)):
        content = content_gen.next()
        style = style_gen.next()
        
        history = model.train_on_batch(content, {"transform_output": style, "block5_pool": dummy_y})
        if i % 5 == 0:
            print("\repoch: %d, loss: %d" % (i, np.array(history).mean()), end="")
            total_loss.append(history)
        
        if i % 50 == 0:
            result = net.predict(np.expand_dims(content[0], axis=0))[0]
            show_without_deprocess(result, img_width, img_height, save=True, iterate=i, style = style_name)

    end_time = time.time() - start_time
    print("\n Consumed time", end_time/3600)
    net.save_weights('models/fast_style_transfer/pretrained/' + style_name+'_weights.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a distallated model')
    
    parser.add_argument('--path', '-p', type=str, required=True,
                        help='training images path')
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file path without extension')
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--content_weight', default=1.0, type=float)
    parser.add_argument('--style_weight', default=4.0, type=float)
    parser.add_argument('--image_size', default=256, type=int)

    args = parser.parse_args()
    main(args)
