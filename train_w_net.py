from w_net_v11 import get_unet
from data_loader import get_data_generators
from keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

    


def main(args):
    img_rows = 128
    img_cols = 832/2
    batch_size = 1
    n_epochs = 100
    models_folder = 'models'
    model_name = 'w_net_V12'
    model_path = os.path.join(models_folder, model_name)
    train = True

    if train:
    	train_generator, val_generator, training_samples, val_samples = get_data_generators(train_folder='train',
                                                                                        val_folder='validation',
                                                                                        img_rows=img_rows,
                                                                                        img_cols=img_cols,
                                                                                        batch_size=batch_size)

    	print('found {} training samples and {} validation samples'.format(training_samples, val_samples))
    	print('...')
    	print('building model...')

    	w_net, disp_map_model = get_unet(img_rows=img_rows, img_cols=img_cols, lr=1e-4)

    	print('saving model to {}...'.format(model_path))
    	model_yaml = w_net.to_yaml()
    	with open(model_path + ".yaml", "w") as yaml_file:
        	yaml_file.write(model_yaml)

    	print('begin training model, {} epochs...'.format(n_epochs))
    	for epoch in range(n_epochs):

        	print('epoch {} \n'.format(epoch))
		print('Validation steps {} \n'.format(val_samples//batch_size))
        	model_path = os.path.join(models_folder, model_name + '_epoch_{}'.format(epoch))
        	w_net.fit_generator(train_generator,
                            steps_per_epoch=training_samples // batch_size,
                            epochs=1,
                            validation_data=val_generator,
                            validation_steps=val_samples // batch_size,
                            verbose=1,
                            callbacks=[TensorBoard(log_dir='/tmp/deepdepth'),
                                       ModelCheckpoint(model_path + '.h5', monitor='loss',
                                                       verbose=0,
                                                       save_best_only=False,
                                                       save_weights_only=True,
                                                       mode='auto', period=1)])
    else:
	data_generator, _ , _, _ = get_data_generators('validation/',
                                                     'validation/', 
                                                     batch_size=1,
                                                    shuffle=False, img_rows=img_rows, img_cols=img_cols)

	w_net, disp_maps_forward = get_unet(img_rows=img_rows, img_cols=img_cols)
	weight_path = model_path + '_epoch_99.h5'
	w_net.load_weights(weight_path)
	images = []
	depthmaps = []
	for i in tqdm(range(12)):
    		dat = data_generator.next()

    		disparity_map_left, disparity_map_right = disp_maps_forward.predict(dat[0][0:10])

    		depthMap_left = np.zeros(disparity_map_left[0,...,0].shape)
    		for i_disp, disp in zip(range(-16,16),np.rollaxis(disparity_map_left[0,...],2)):
        		depthMap_left += disp*i_disp

    		depthMap_right = np.zeros(disparity_map_right[0,...,0].shape)
    		for i_disp, disp in zip(range(-16,16),np.rollaxis(disparity_map_right[0,...],2)):
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
