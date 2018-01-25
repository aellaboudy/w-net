import matplotlib
matplotlib.use('TkAgg')

from w_net_v11 import get_unet
from data_loader import get_data_generators
from keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf    


def main(args):
    img_rows = 128
    img_cols = 832/2
    models_folder = 'models'
    model_name = 'w_net_V12'
    model_path = os.path.join(models_folder, model_name)
    data_generator, _ , _, _ = get_data_generators('/home/ubuntu/data/stereoimages/images/test/',
                                                     '/home/ubuntu/data/stereoimages/images/test/', 
                                                     batch_size=1,
                                                    shuffle=False, img_rows=img_rows, img_cols=img_cols)


    with tf.device("/cpu:0"):
    	w_net, disp_maps_forward = get_unet(img_rows=img_rows, img_cols=img_cols, lr=1e-7)
    	
    	w_net.compile(optimizer=Adam(lr=1e-7), loss='mean_absolute_error', loss_weights=[1.,1.,1.,1.,1.,1.])


    	weight_path = model_path + '.h5'
    	w_net.load_weights(weight_path)
    	images = []
    	depthmaps = []
    	for i in tqdm(range(12)):
    		dat = data_generator.next()
                left_reconstruct, right_reconstruct, output_reconstruct,output_consistency, weighted_gradient_left, weighted_gradient_right = w_net.predict(dat[0][0:10])
    		depth_left, depth_right = disp_maps_forward.predict(dat[0][0:10])

    		plt.figure(figsize=(15,5))
    		
                print (depth_left)
                print (depth_right)
                plt.subplot(1,2,2)
                plt.imshow(dat[0][0,:,:img_cols,:])
    		plt.axis('off')
    		plt.subplot(1,2,1)
    		plt.imshow(depth_left[0,...], cmap=plt.cm.jet)
    		plt.axis('off')
                

                #plt.subplot(3,2,4)
                #plt.imshow(dat[0][0,:,img_cols:,:])
                #plt.axis('off')
                #plt.subplot(3,2,3)
                #plt.imshow(depth_right[0,...], cmap=plt.cm.jet)
                #plt.axis('off')

                #plt.subplot(3,2,5)
                #plt.imshow(output_reconstruct[0,...])
                #plt.axis('off')
                #plt.subplot(3,2,6)
                #plt.imshow(output_consistency[0,...])
                #plt.axis('off')
    		plt.show()
			




if __name__ == '__main__':
    main(None)
