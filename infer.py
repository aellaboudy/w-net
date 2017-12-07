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
    img_rows = 128/2
    img_cols = 832/4
    models_folder = 'models'
    model_name = 'w_net_V12'
    model_path = os.path.join(models_folder, model_name)
    data_generator, _ , _, _ = get_data_generators('/home/amel/data/stereoimages/images/test/',
                                                     '/home/amel/data/stereoimages/images/test/', 
                                                     batch_size=1,
                                                    shuffle=False, img_rows=img_rows, img_cols=img_cols)

    w_net, disp_maps_forward = get_unet(img_rows=img_rows, img_cols=img_cols, lr=1e-7)
    	
    w_net.compile(optimizer=Adam(lr=1e-7), loss='mean_absolute_error', loss_weights=[1.,1.,0.001,0.001])


    weight_path = model_path + '.h5'
    w_net.load_weights(weight_path)
    images = []
    depthmaps = []
    for i in tqdm(range(12)):
    	dat = data_generator.next()

    	disparity_map_left, disparity_map_right = disp_maps_forward.predict(dat[0][0:10])

    	depthMap_left = np.zeros(disparity_map_left[0,...,0].shape)
    	for i_disp, disp in zip(range(0,128),np.rollaxis(disparity_map_left[0,...],2)):
        	depthMap_left += disp*i_disp

    	depthMap_right = np.zeros(disparity_map_right[0,...,0].shape)
    	for i_disp, disp in zip(range(0,-128),np.rollaxis(disparity_map_right[0,...],2)):
        	depthMap_right += disp*i_disp
        
    	plt.figure(figsize=(15,5))
    	plt.subplot(1,2,2)
    	plt.imshow(dat[0][0,:,:img_cols,:])
    	plt.axis('off')
    	plt.subplot(1,2,1)
    	plt.imshow(depthMap_left, cmap=plt.cm.jet)
    	plt.axis('off')
    	plt.show()
			




if __name__ == '__main__':
    main(None)
