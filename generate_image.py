import os
import argparse
import models.fast_transfer as fast_transfer
import models.style_swap_transfer as style_swap_transfer
import models.mask_style_transfer as mask_style_transfer
import base64, json, sys
import style_names
import gc
import models.src.nets as nets
from keras.preprocessing import image

from models.src.layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from models.src.loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
import models.src.nets as nets
import numpy as np
import time
from PIL import Image
from scipy.misc import imread, imresize, imsave, fromimage, toimage

def get_ratio(image):
    img = Image.fromarray(image).convert('RGB')
    img_WIDTH, img_HEIGHT = img.size
    aspect_ratio = float(img_HEIGHT) / img_WIDTH
    return aspect_ratio

def save_without_deprocess(x, img_width, img_height, save=True, name='', iterate=0, style="udnie"):

    x = x.reshape((img_width, img_height, 3))
    img = np.clip(x, 0, 255).astype('uint8')
    
    # 取得原圖比例
    aspect_ratio = get_ratio(img)  
    img_ht = int(img_width * aspect_ratio)
    #print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
    img = imresize(img, (img_width, img_ht), interp='bilinear')
    im = toimage(img)
    if save:
        dir_name = "output/{}/0".format(style)
        filename = dir_name + "/{}".format(name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        imsave(filename, im)
    else:
        plt.imshow(im)

def main(args):
    weights = os.listdir("./models/fast_style_transfer/original_pretrained/")
    styles = [filename.split("_weights")[0] for filename in weights]
    if len(styles) == 0:
        print("You have no original pretrained model, please check the link on github to download them")
    else:
        print("You have {} pretrained models can be used to produce images".format(len(styles)))
    
    datagen = image.ImageDataGenerator()

    image_path = args.path
    epoch = 82783
    batch_size=1
    img_width = img_height = args.image_size

    content_gen = datagen.flow_from_directory(image_path,
                                            shuffle=False,
                                                target_size=(img_width, img_height),
                                                batch_size=batch_size,
                                                class_mode=None)

    model = nets.image_transform_net(img_width,img_height)
    start_time = time.time()

    for style in styles:
        print("\nProcessing {}".format(style))
        single_model_start_time = time.time()
        model = nets.image_transform_net(img_width,img_height)
        model.load_weights("./models/fast_style_transfer/original_pretrained/"+style+'_weights.h5')
        for i in range(epoch):
            content = content_gen.next()

            result = model.predict(content)

            image_name = content_gen.filenames[i][2:]

            if i % 10 == 0:
                print("\repoch: %d" %(i), end="")

            save_without_deprocess(result, img_width, img_height, save=True, name=image_name, iterate=i, style = style)
        print("\nProcessing {} finish in {}".format(style, time.time() - single_model_start_time)) 
        gc.collect()

    end_time = time.time() - start_time
    print("\nTotal Time cost: ", end_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Using pretrained model to generate stylize images')
    
    parser.add_argument('--path', '-p', type=str, required=True,
                        help='training images path')
    parser.add_argument('--image_size', default=224, type=int)
    args = parser.parse_args()
    main(args)