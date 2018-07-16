from __future__ import print_function

from keras import backend as K
from keras.layers import Dropout
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Lambda
from keras.layers.convolutional import SeparableConv2D
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
from bilinear_sampler import *
import tensorflow as tf


K.set_image_data_format('channels_last')  # TF dimension ordering in this code



#Convert this to do bilinear sampling instead. Input should be an image and an output of Depth layer.
class Selection(Layer):
    def __init__(self,  **kwargs):
        # if none, initialize the disparity levels as described in deep3d

        super(Selection, self).__init__(**kwargs)


    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Selection` layer should be called '
                             'on a list of 2 inputs.')

    def call(self, inputs):
	return bilinear_sampler_1d_h(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Gradient(Layer):
    def __init__(self, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        super(Gradient, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        pass

    def call(self, inputs):
        dinputs_dx = inputs - K.concatenate([K.zeros_like(inputs[..., :1, :, :]), inputs[..., :-1, :, :]], axis=1)
        #dinputs_dx_1 = inputs - K.concatenate([inputs[..., 1:, :,:], K.zeros_like(inputs[..., :1, :,:])], axis=1)

        dinputs_dy = inputs - K.concatenate([K.zeros_like(inputs[..., :1,:]), inputs[..., :-1,:]], axis=2)
        #dinputs_dy_1 = inputs - K.concatenate([inputs[..., 1:,:], K.zeros_like(inputs[..., :1,:])], axis=2)

	#dinput_dx = Lambda(lambda x : K.mean(K.abs(x[0] + x[1]),axis=3)) ([dinputs_dx_0, dinputs_dx_1])
	#dinput_dy = Lambda(lambda x : K.mean(K.abs(x[0] + x[1]),axis=3)) ([dinputs_dy_0, dinputs_dy_1])
	

	return [dinputs_dx[:,1:,1:] , dinputs_dy[:,1:,1:]]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1] - 1 , input_shape[2] - 1, input_shape[3]), (input_shape[0], input_shape[1] - 1, input_shape[2] - 1, input_shape[3])]
	#return [(None,None,None), (None,None,None)]



    def compute_mask(self, input, input_mask=None):
        return [None, None]

#Convert this to do a soft argmin instead to produce a H X W depth map...already DONE!!
class Depth(Layer):
    def __init__(self, disparity_levels=None, **kwargs):
        # if none, initialize the disparity levels as described in deep3d
        if disparity_levels is None:
            disparity_levels = range(-3, 9, 1)

        # if none, initialize the disparity levels as described in deep3d
        super(Depth, self).__init__(**kwargs)

        self.disparity_levels = disparity_levels

    def build(self, input_shape):
        # Used purely for shape validation.
        pass

    def call(self, disparity):

	depth = []
        for n, disp in enumerate(self.disparity_levels):
            depth += [disparity[..., n] * disp]
        
        depth = K.stack(depth, axis=0)
	return K.sum(depth, axis=0, keepdims=False)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def get_unet(img_rows, img_cols, lr=1e-4, dl = 50/2):
    inputs = Input((img_rows, 2 * img_cols, 3))  # 2 channels: left and right images

	
    # split input left/right wise
    left_input_image = Lambda(lambda x: x[..., :img_cols, :])(inputs)
    right_input_image = Lambda(lambda x: x[..., img_cols:, :])(inputs)

    concatenated_images = concatenate([left_input_image, right_input_image], axis=3)

    conv1 = SeparableConv2D(dl*2, (3, 3), activation='relu', padding='same')(concatenated_images)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(dl*2, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = SeparableConv2D(dl*4, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = SeparableConv2D(dl*4, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = SeparableConv2D(dl*8, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = SeparableConv2D(dl*8, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = SeparableConv2D(dl*16, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = SeparableConv2D(dl*16, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = SeparableConv2D(dl*32, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = SeparableConv2D(dl*32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(dl*16, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(dl*16, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SeparableConv2D(dl*16, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(rate=0.4)(conv6)

    up7 = concatenate([Conv2DTranspose(dl*8, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(dl*8, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SeparableConv2D(dl*8, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(rate=0.4)(conv7)

    up8 = concatenate([Conv2DTranspose(dl*4, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(dl*4, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SeparableConv2D(dl*4, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(rate=0.4)(conv8)

    up9 = concatenate([Conv2DTranspose(dl*4, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(dl*4, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SeparableConv2D(dl*4, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(rate=0.4)(conv9)

    # split into left/right disparity maps

    left_disparity_level_4 = Conv2DTranspose(dl*2, (16, 16), strides=(16, 16), padding='same')(
        Lambda(lambda x: x[..., dl*8:])(pool4))
    right_disparity_level_4 = Conv2DTranspose(dl*2, (16, 16), strides=(16, 16), padding='same')(
        Lambda(lambda x: x[..., :dl*8])(pool4))


    left_disparity_level_3 = Conv2DTranspose(dl*2, (8, 8), strides =(8, 8), padding='same')(
        Lambda(lambda x: x[..., dl*4:])(pool3))
    right_disparity_level_3 = Conv2DTranspose(dl*2, (8, 8), strides=(8, 8), padding='same')(
        Lambda(lambda x: x[..., :dl*4])(pool3))

    left_disparity_level_2 = Conv2DTranspose(dl*2, (4, 4), strides=(4, 4), padding='same')(
        Lambda(lambda x: x[..., dl*2:])(pool2))
    right_disparity_level_2 = Conv2DTranspose(dl*2, (4, 4), strides=(4, 4), padding='same')(
        Lambda(lambda x: x[..., :dl*2])(pool2))

    left_disparity_level_1 = Lambda(lambda x: x[..., :dl*2])(conv9)
    right_disparity_level_1 = Lambda(lambda x: x[..., dl*2:])(conv9)

    left_disparity = Lambda(lambda x: K.mean(K.stack([xi for xi in x]), axis=0))([left_disparity_level_1,
                                                                                  left_disparity_level_2,
                                                                                  left_disparity_level_3,
                                                                                  left_disparity_level_4])

    right_disparity = Lambda(lambda x: K.mean(K.stack([xi for xi in x]), axis=0))([right_disparity_level_1,
                                                                                   right_disparity_level_2,
                                                                                   right_disparity_level_3,
                                                                                   right_disparity_level_4])

    # use a softmax activation on the conv layer output to get a probabilistic disparity map
    left_disparity = SeparableConv2D(dl*2, (3, 3), activation='softmax', padding='same')(left_disparity)

    right_disparity = SeparableConv2D(dl*2, (3, 3), activation='softmax', padding='same')(right_disparity)


    left_disparity_levels = range(0, dl*2, 1)
    right_disparity_levels = range(0, -dl*2, -1)

    depth_left = Depth(disparity_levels=left_disparity_levels)(left_disparity)
    depth_right = Depth(disparity_levels=right_disparity_levels)(right_disparity)
   
     
    depth_left_gradient_x, depth_left_gradient_y = Gradient()(Lambda (lambda x: K.expand_dims(x,axis=3)) (depth_left))
    depth_right_gradient_x, depth_right_gradient_y = Gradient()(Lambda (lambda x: K.expand_dims(x,axis=3)) (depth_right))
    #depth_left_gradient_x, _ = Gradient()(depth_left_gradient_x)
    #_, depth_left_gradient_y = Gradient()(depth_left_gradient_y)
    #depth_right_gradient_x,_ = Gradient()(depth_right_gradient_x)
    #_, depth_right_gradient_y = Gradient()(depth_right_gradient_y)


    right_reconstruct_im = Selection()([left_input_image, depth_left])

    left_reconstruct_im = Selection()([right_input_image, depth_right])

    right_consistency_im = Selection()([left_reconstruct_im, depth_left])
    left_consistency_im = Selection()([right_reconstruct_im, depth_right])	

    # concatenate left and right images along the channel axis
    output_reconstruct = concatenate([left_reconstruct_im, right_reconstruct_im], axis=2)
    
    output_consistency = concatenate([left_consistency_im, right_consistency_im], axis=2)

    #left_input_gray = Lambda(lambda x: K.squeeze(tf.image.rgb_to_grayscale(x),axis=3))(left_input_image)
    #right_input_gray = Lambda(lambda x: K.squeeze(tf.image.rgb_to_grayscale(x),axis=3))(right_input_image)
    image_left_gradient_x, image_left_gradient_y = Gradient()(left_input_image)
    image_right_gradient_x, image_right_gradient_y  = Gradient()(right_input_image)
    #image_left_gradient_x, _ = Gradient()(image_left_gradient_x)
    #_, image_left_gradient_y = Gradient()(image_left_gradient_y)
    #image_right_gradient_x,_ = Gradient()(image_right_gradient_x)
    #_, image_right_gradient_y = Gradient()(image_right_gradient_y)

    #left_reconstruct_gray = Lambda(lambda x: K.squeeze(tf.image.rgb_to_grayscale(x),axis=3))(left_reconstruct_im)
    #right_reconstruct_gray = Lambda(lambda x: K.squeeze(tf.image.rgb_to_grayscale(x),axis=3))(right_reconstruct_im)
    #output_gradient_left = Lambda(lambda x: K.abs(x[0] - x[2]) + K.abs(x[1] - x[3])) (Gradient()(left_reconstruct_gray) + Gradient()(left_input_gray))
    #output_gradient_right = Lambda(lambda x: K.abs(x[0] - x[2]) + K.abs(x[1] - x[3])) (Gradient()(right_reconstruct_gray) + Gradient()(right_input_gray))
	
    weighted_gradient_left = Lambda(lambda x: K.abs(x[0]) * K.exp(-K.mean(K.abs(x[1]),axis=3,keepdims=True)) + K.abs(x[2]) * K.exp(-K.mean(K.abs(x[3]),axis=3,keepdims=True)))([depth_left_gradient_x, image_left_gradient_x, depth_left_gradient_y, image_left_gradient_y])
    weighted_gradient_right = Lambda(lambda x: K.abs(x[0]) * K.exp(-K.mean(K.abs(x[1]),axis=3,keepdims=True)) + K.abs(x[2]) * K.exp(-K.mean(K.abs(x[3]),axis=3,keepdims=True)))([depth_right_gradient_x, image_right_gradient_x, depth_right_gradient_y, image_right_gradient_y])


    model = Model(inputs=[inputs], outputs=[left_reconstruct_im,right_reconstruct_im,output_reconstruct,output_consistency, weighted_gradient_left, weighted_gradient_right,depth_left,depth_right])


    disp_map_model = Model(inputs=[inputs], outputs=[depth_left, depth_right])

    # we use L1 type loss as it has been shown to work better for that type of problem in the deep3d paper
    # (https://arxiv.org/abs/1604.03650)
    #model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error', loss_weights=[1.,1.,0.001,0.001])
    model.summary()


    return model, disp_map_model
