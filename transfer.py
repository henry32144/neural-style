from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
from PIL import Image
from io import BytesIO

import numpy as np
import time
import argparse
import warnings
import sys
import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.backend.tensorflow_backend import set_session
"""
Neural Style Transfer with Keras 2.0.5

Based on:
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
https://github.com/titu1994/Neural-Style-Transfer

Contains few improvements suggested in the paper Improving the Neural Algorithm of Artistic Style
(http://arxiv.org/abs/1605.04603).

-----------------------------------------------------------------------------------------------------------------------
"""

THEANO_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

TH_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def transfer(base_image, syle_image_path):
    def str_to_bool(v):
        return v.lower() in ("true", "yes", "t", "1")

    ''' Arguments '''
    ##args = parser.parse_args()
    IMG_HEIGHT = 400
    IMG_WIDTH = 400
    chosen_model = 'vgg16'

    # Need to pass image path
    base_image_path = base_image
    style_image_path = syle_image_path


    # these are the weights of the different loss components
    content_weight = 0.1
    style_weight = 1.0
    total_variation_weight = 8.5e-5

    # 取得輸入圖片的長寬比例
    def get_ratio(image):
        img = np.asarray(Image.open(image).convert('RGB')).astype('float')
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH
        return aspect_ratio

    # 預處理輸入的圖片
    def preprocess_image(image_path):
        mode = "RGB"
        img = imread(image_path, mode=mode)

        img = imresize(img,(IMG_WIDTH, IMG_HEIGHT)).astype('float32')


        # RGB -> BGR
        img = preprocess_input(img)
        
        # 檢查當前keras使用的後端引擎，如果是Theano的話需要轉換圖片陣列格式
        # Theano 格式 : (channels, rows, cols). ex: In our case - (3, 400, 400) 
        # Tensorflow 格式 : (rows, cols, channels). ex: In our case - (400, 400, 3)
        if K.image_dim_ordering() == "th":
            img = img.transpose((2, 0, 1)).astype('float32')

        # 將圖片陣列多拓展一個存放圖片數量的維度，ex: 原本是(400, 400, 3), 變成(1, 400, 400, 3)
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        return img

    # 後處理要輸出的圖片，將處理完的矩陣轉回去照片的格式
    def deprocess_image(x):
        if K.image_dim_ordering() == "th":
            x = x.reshape((3, IMG_WIDTH, IMG_HEIGHT))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((IMG_WIDTH, IMG_HEIGHT, 3))

        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        
        # BGR -> RGB
        x = x[:, :, ::-1]

        # 將陣列的值的範圍縮回 0~255，因為處理的結果有可能出現超過這個範圍的數字
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    # 建立存放經過預處理後的照片的變數
    base_image = K.variable(preprocess_image(base_image_path))
    style_image = K.variable(preprocess_image(style_image_path))

    if K.image_dim_ordering() == 'th':
        combination_image = K.placeholder((1, 3, IMG_WIDTH, IMG_HEIGHT))
    else:
        combination_image = K.placeholder((1, IMG_WIDTH, IMG_HEIGHT, 3))

    image_tensors = [base_image, style_image, combination_image]
    # nb = number
    nb_tensors = len(image_tensors)
    nb_style_images = 1 

    # 將存放照片的陣列壓成一串Tensor(張量)，用於當作模型的輸入
    input_tensor = K.concatenate(image_tensors, axis=0)

    if K.image_dim_ordering() == "th":
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT,None)
    else:
        input_shape = (None, IMG_WIDTH, IMG_HEIGHT, 3)

    #load model
    model = VGG16(include_top=False,weights='imagenet', input_tensor=input_tensor)

    print(chosen_model)

    if K.backend() == 'tensorflow' and K.image_dim_ordering() == "th":
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image dimension ordering convention '
                      '(`image_dim_ordering="th"`). '
                      'For best performance, set '
                      '`image_dim_ordering="tf"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
        convert_all_kernels_in_model(model)

    print('Model loaded.')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    def gram_matrix(x):
        
        # ndim: 取得張量的階數
        assert K.ndim(x) == 3
        if K.image_dim_ordering() == "th":
            features = K.batch_flatten(x)
        else:
            # batch_flatten: 將一個n階張量轉變為2階張量，其第一維度保留不變
            # permute_dimensions: 按照给定的模式重排一个張量的軸
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

        # dot: 矩陣乘法
        # transpose: 矩陣轉置
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(style, combination):
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3

        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = IMG_WIDTH * IMG_HEIGHT
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


    def content_loss(base, combination):

        return 0.5 * K.sum(K.square(content_weight * (combination - base)))


    # 這個loss幫助圖片的細節有連貫性
    # 詳細: https://blog.csdn.net/afgh2587849/article/details/6401181
    def total_variation_loss(x):
        assert K.ndim(x) == 4
        # square: 平方
        if K.image_dim_ordering() == 'th':
            a = K.square(x[:, :, :IMG_WIDTH - 1, :IMG_HEIGHT - 1] - x[:, :, 1:, :IMG_HEIGHT - 1])
            b = K.square(x[:, :, :IMG_WIDTH - 1, :IMG_HEIGHT - 1] - x[:, :, :IMG_WIDTH - 1, 1:])
        else:
            a = K.square(x[:, :IMG_WIDTH - 1, :IMG_HEIGHT - 1, :] - x[:, 1:, :IMG_HEIGHT - 1, :])
            b = K.square(x[:, :IMG_WIDTH - 1, :IMG_HEIGHT - 1, :] - x[:, :IMG_WIDTH - 1, 1:, :])
        # sum: 總和
        # pow: 乘上次方
        return K.sum(K.pow(a + b, 1.25))

    feature_layers = ['block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool',
                     'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 
                     'block4_conv3','block4_pool','block5_conv1', 'block5_conv2', 'block5_conv3']

    loss = K.variable(0.)

    # 取出'block5_conv2'這一層得到的特徵值為content feature
    layer_features = outputs_dict['block5_conv2']
    
    # 張量中的第0個元素是content image，格式是這樣 [base_image(0), style_image(1), combination_image(2)]
    # 冒號語法的意思是從頭到尾，也就是該維度的元素全部選取的意思
    base_image_features = layer_features[0, :, :, :]

    # 張量中的第2個元素是combination image， [base_image(0), style_image(1), combination_image(2)]
    # nb_tensors的值是3
    combination_features = layer_features[nb_tensors - 1, :, :, :]

    # 將loss值加上乘完權重的content loss
    loss += content_loss(base_image_features,combination_features)
    
    nb_layers = len(feature_layers) - 1

    channel_index = 1 if K.image_dim_ordering() == "th" else -1

    # Improvement 3 : Chained Inference without blurring
    for i in range(len(feature_layers) - 1):
        
        # 當前feature layer
        layer_features = outputs_dict[feature_layers[i]]
        combination_features = layer_features[nb_tensors - 1, :, :, :]
        style_reference_features = layer_features[1, :, :, :]

        sl1 = style_loss(style_reference_features, combination_features)

        # 下一個feature layer
        layer_features = outputs_dict[feature_layers[i + 1]]
        combination_features = layer_features[nb_tensors - 1, :, :, :]
        style_reference_features = layer_features[1, :, :, :]
        
        sl2 = style_loss(style_reference_features, combination_features)

        sl = sl1 - sl2
        
        # 將loss值加上乘完權重的style loss
        loss += (style_weight / (2 ** (nb_layers - (i + 1)))) * sl

    loss += total_variation_weight * total_variation_loss(combination_image)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)

    outputs = [loss]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([combination_image], outputs)


    def eval_loss_and_grads(x):
        if K.image_dim_ordering() == 'th':
            x = x.reshape((1, 3, IMG_WIDTH, IMG_HEIGHT))
        else:
            x = x.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))
        outs = f_outputs([x])
        # outs 第0個為loss, 第1個gradient
        loss_value = outs[0]
        
        # 檢查梯度值，如果只有1個就壓平那一個;如果超過1個，就壓平全部
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


    # this Evaluator class makes it possible
    # to compute loss and gradients in one pass
    # while retrieving them via two separate functions,
    # "loss" and "grads". This is done because scipy.optimize
    # requires separate functions for loss and gradients,
    # but computing them separately would be inefficient.
    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values


    evaluator = Evaluator()

    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss

    x = preprocess_image(base_image_path)

    num_iter = 10
    prev_min_val = -1

    improvement_threshold = float(0.0)
    output = BytesIO()

    for i in range(num_iter):
        print("Starting iteration %d of %d" % ((i + 1), num_iter))
        start_time = time.time()

        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)

        if prev_min_val == -1:
            prev_min_val = min_val

        improvement = (prev_min_val - min_val) / prev_min_val * 100

        print("Current loss value:", min_val, " Improvement : %0.3f" % improvement, "%")
        prev_min_val = min_val
        # save current generated image
        img = deprocess_image(x.copy())
        
        aspect_ratio = get_ratio(base_image_path)
        img_ht = int(IMG_WIDTH * aspect_ratio)
        print("Rescaling Image to (%d, %d)" % (IMG_WIDTH, img_ht))
        img = imresize(img, (IMG_WIDTH, img_ht), interp='bilinear')
        
        end_time = time.time()
        print("Iteration %d completed in %ds" % (i + 1, end_time - start_time))

        if i >= num_iter - 1:
            print(type(img), file=sys.stdout)
            im = toimage(img)
            print(type(im), file=sys.stdout)
            im.save(output, format='JPEG')
            return output.getvalue()
        
        if improvement_threshold is not 0.0:
            if improvement < improvement_threshold and improvement is not 0.0:
                print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." %
                      (improvement, improvement_threshold))
                exit()
