from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import add
from keras.engine import InputSpec
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Deconvolution2D, Conv2D,UpSampling2D,Cropping2D
from keras.applications.vgg16 import preprocess_input
from keras.layers.advanced_activations import LeakyReLU
from keras.applications.imagenet_utils import  preprocess_input

from tensorflow.python.layers import utils

from models.src.VGG16 import VGG16

import numpy as np
import tensorflow as tf



class InputNormalize(Layer):
    def __init__(self, **kwargs):
        super(InputNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self,input_shape):
        return input_shape

    def call(self, x, mask=None):
        #x = (x - 127.5)/ 127.5
        return x/255.

class InstanceNormalize(Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalize, self).__init__(**kwargs)
        self.epsilon = 1e-3
            

    def call(self, x, mask=None):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, self.epsilon)))

                                                 
    def compute_output_shape(self,input_shape):
        return input_shape


class Denormalize(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)
    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''

        return (x + 1) * 127.5

    def compute_output_shape(self,input_shape):
        return input_shape


class VGGNormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.
    '''

    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        # No exact substitute for set_subtensor in tensorflow
        # So we subtract an approximate value       
        
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]       
        x -= 120
        #img_util.preprocess_image(style_image_path, img_width, img_height)
        return x
   

    def compute_output_shape(self,input_shape):
        return input_shape




class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 2:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
            elif len(padding) == 4:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))

        if dim_ordering not in {'tf'}:
            raise ValueError('dim_ordering must be in {tf}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)] 


    def call(self, x, mask=None):
        top_pad=self.top_pad
        bottom_pad=self.bottom_pad
        left_pad=self.left_pad
        right_pad=self.right_pad        
        
        paddings = [[0,0],[left_pad,right_pad],[top_pad,bottom_pad],[0,0]]

        
        return tf.pad(x,paddings, mode='REFLECT', name=None)

    def compute_output_shape(self,input_shape):
        if self.dim_ordering == 'tf':
            rows = input_shape[1] + self.top_pad + self.bottom_pad if input_shape[1] is not None else None
            cols = input_shape[2] + self.left_pad + self.right_pad if input_shape[2] is not None else None

            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
            
    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))     
    
    
class UnPooling2D(UpSampling2D):
    def __init__(self, size=(2, 2)):
        super(UnPooling2D, self).__init__(size)

  
    def call(self, x, mask=None):
        shapes = x.get_shape().as_list() 
        w = self.size[0] * shapes[1]
        h = self.size[1] * shapes[2]
        return tf.image.resize_nearest_neighbor(x, (w,h))

class TanhNormalize(Layer):

    def __init__(self, **kwargs):
        super(TanhNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        # No exact substitute for set_subtensor in tensorflow
        # So we subtract an approximate value       
        
        # 'RGB'->'BGR'
        x = (x + 1) * (255.0 / 2)
        return x
   

    def compute_output_shape(self,input_shape):
        return input_shape


def conv_bn_relu(nb_filter, nb_row, nb_col,stride):   
    def conv_func(x):
        x = Conv2D(nb_filter, (nb_row, nb_col), strides=stride,padding='same')(x)
        x = BatchNormalization()(x)
        #x = LeakyReLU(0.2)(x)
        x = Activation("relu")(x)
        return x
    return conv_func

def conv_in_relu(nb_filter, nb_row, nb_col,stride):   
    def conv_func(x):
        x = Conv2D(nb_filter, (nb_row, nb_col), strides=stride,padding='same')(x)
        x = InstanceNormalize()(x)
        x = Activation("relu")(x)
        return x
    return conv_func

#https://keunwoochoi.wordpress.com/2016/03/09/residual-networks-implementation-on-keras/
def res_conv(nb_filter, nb_row, nb_col,stride=(1,1)):
    def _res_func(x):
        identity = Cropping2D(cropping=((2,2),(2,2)))(x)

        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(a)
        y = BatchNormalization()(a)

        return  add([identity, y])

    return _res_func

def res_in_conv(nb_filter, nb_row, nb_col,stride=(1,1)):
    def _res_func(x):
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same')(x)
        a = InstanceNormalize()(a)
        a = Activation("relu")(a)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same')(a)
        y = InstanceNormalize()(a)
        return  add([x, y])
    return _res_func

def dconv_bn_nolinear(nb_filter, nb_row, nb_col,stride=(2,2),activation="relu"):
    def _dconv_bn(x):
        x = UnPooling2D(size=stride)(x)
        x = ReflectionPadding2D(padding=stride)(x)
        x = Conv2D(nb_filter, (nb_row, nb_col), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    return _dconv_bn


def wct_style_swap(x, alpha=0.8, patch_size=3, stride=1, eps=1e-8):
    '''Modified Whiten-Color Transform that performs style swap on whitened content/style encodings before coloring
       Assume that content/style encodings have shape 1xHxWxC
    '''  
    content = K.expand_dims(x[0], 0)
    style = K.expand_dims(x[1], 0)
    
    content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
    style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

    Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
    Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

    # CxHxW -> CxH*W
    content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
    style_flat = tf.reshape(style_t, (Cs, Hs*Ws))

    # Content covariance
    mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
    fc = content_flat - mc
    fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps

    # Style covariance
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps

    # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
    with tf.device('/cpu:0'):  
        Sc, Uc, _ = tf.svd(fcfc)
        Ss, Us, _ = tf.svd(fsfs)

    ## Uncomment to perform SVD for content/style with np in one call
    ## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
    # @jit
    # def np_svd(content, style):
    #     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
    #     Uc, Sc, _ = np.linalg.svd(content)
    #     Us, Ss, _ = np.linalg.svd(style)
    #     return Uc, Sc, Us, Ss
    # Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])
    
    k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
    k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

    ### Whiten content
    Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))

    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)

    # Reshape before passing to style swap, CxH*W -> 1xHxWxC
    whiten_content = tf.expand_dims(tf.transpose(tf.reshape(fc_hat, [Cc,Hc,Wc]), [1,2,0]), 0)

    ### Whiten style before swapping
    Ds = tf.diag(tf.pow(Ss[:k_s], -0.5))
    whiten_style = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fs)
    # Reshape before passing to style swap, CxH*W -> 1xHxWxC
    whiten_style = tf.expand_dims(tf.transpose(tf.reshape(whiten_style, [Cs,Hs,Ws]), [1,2,0]), 0)

    ### Style swap whitened encodings
    #ss_feature = ori_style_swap_layer(whiten_content, whiten_style, patch_size, stride)
    
    ###############################################
    nC = tf.shape(whiten_style)[-1]  # Num channels of input content feature and style-swapped output

    ### Extract patches from style image that will be used for conv/deconv layers
    style_patches = tf.extract_image_patches(whiten_style, [1,patch_size,patch_size,1], [1,stride,stride,1], [1,1,1,1], 'VALID')
    before_reshape = tf.shape(style_patches)  # NxRowsxColsxPatch_size*Patch_size*nC
    style_patches = tf.reshape(style_patches, [before_reshape[1]*before_reshape[2],patch_size,patch_size,nC])
    style_patches = tf.transpose(style_patches, [1,2,3,0])  # Patch_sizexPatch_sizexIn_CxOut_c

    # Normalize each style patch
    style_patches_norm = tf.nn.l2_normalize(style_patches, dim=3)

    # Compute cross-correlation/nearest neighbors of patches by using style patches as conv filters
    ss_enc = tf.nn.conv2d(whiten_content,
                          style_patches_norm,
                          [1,stride,stride,1],
                          'VALID')

    # For each spatial position find index of max along channel/patch dim  
    ss_argmax = tf.argmax(ss_enc, axis=3)
    encC = tf.shape(ss_enc)[-1]  # Num channels in intermediate conv output, same as # of patches
    
    # One-hot encode argmax with same size as ss_enc, with 1's in max channel idx for each spatial pos
    ss_oh = tf.one_hot(ss_argmax, encC, 1., 0., 3)

    # Calc size of transposed conv out
    deconv_out_H = utils.deconv_output_length(tf.shape(ss_oh)[1], patch_size, 'valid', stride)
    deconv_out_W = utils.deconv_output_length(tf.shape(ss_oh)[2], patch_size, 'valid', stride)
    deconv_out_shape = tf.stack([1,deconv_out_H,deconv_out_W,nC])

    # Deconv back to original content size with highest matching (unnormalized) style patch swapped in for each content patch
    ss_dec = tf.nn.conv2d_transpose(ss_oh,
                                    style_patches,
                                    deconv_out_shape,
                                    [1,stride,stride,1],
                                    'VALID')

    ### Interpolate to average overlapping patch locations
    ss_oh_sum = tf.reduce_sum(ss_oh, axis=3, keep_dims=True)

    filter_ones = tf.ones([patch_size,patch_size,1,1], dtype=tf.float32)
    
    deconv_out_shape = tf.stack([1,deconv_out_H,deconv_out_W,1])  # Same spatial size as ss_dec with 1 channel

    counting = tf.nn.conv2d_transpose(ss_oh_sum,
                                         filter_ones,
                                         deconv_out_shape,
                                         [1,stride,stride,1],
                                         'VALID')

    counting = tf.tile(counting, [1,1,1,nC])  # Repeat along channel dim to make same size as ss_dec

    ss_feature = tf.divide(ss_dec, counting)
    ###############################################
    
    # HxWxC -> CxH*W
    ss_feature = tf.transpose(tf.reshape(ss_feature, [Hc*Wc,Cc]), [1,0])

    ### Color style-swapped encoding with style 
    Ds_sq = tf.diag(tf.pow(Ss[:k_s], 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds_sq), Us[:,:k_s], transpose_b=True), ss_feature)
    fcs_hat = fcs_hat + ms

    ### Blend style-swapped & colored encoding with original content encoding
    blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)
    # CxH*W -> CxHxW
    blended = tf.reshape(blended, (Cc,Hc,Wc))
    # CxHxW -> 1xHxWxC
    blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)

    return blended
