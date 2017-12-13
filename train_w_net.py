from w_net_v11 import get_unet
from data_loader import get_data_generators
from keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam

    


def main(args):
    img_rows = 128
    img_cols = 832/2
    batch_size = 1
    n_epochs = 100
    models_folder = 'models'
    model_name = 'w_net_V12'
    model_path = os.path.join(models_folder, model_name)

    train_generator, val_generator, training_samples, val_samples = get_data_generators(train_folder='/home/ubuntu/data/stereoimages/images/train/',
                                                                                        val_folder='/home/ubuntu/data/stereoimages/images/val/',
                                                                                        img_rows=img_rows,
                                                                                        img_cols=img_cols,
                                                                                        batch_size=batch_size)

    print('found {} training samples and {} validation samples'.format(training_samples, val_samples))
    print('...')
    print('building model...')

    
    w_net, disp_map_model = get_unet(img_rows=img_rows, img_cols=img_cols, lr=1e-7)

    w_net.compile(optimizer=Adam(lr=1e-7), loss='mean_absolute_error', loss_weights=[1.,1.,0.001,0.001])
	
    w_net.load_weights(model_path + '.h5')
    #print('saving model to {}...'.format(model_path))
    #model_yaml = w_net.to_yaml()
    #with open(model_path + ".yaml", "w") as yaml_file:


    print('begin training model, {} epochs...'.format(n_epochs))
    print('Validation steps {} \n'.format(val_samples//batch_size))
    model_path = os.path.join(models_folder, model_name)


    w_net.fit_generator(train_generator,
                            steps_per_epoch=training_samples // batch_size,
                            epochs=n_epochs,
                            validation_data=val_generator,
                            validation_steps=val_samples // batch_size,
                            verbose=1,
                            callbacks=[TensorBoard(log_dir='/tmp/deepdepth'),
                                       ModelCheckpoint(model_path + '.h5', monitor='loss',
                                                       verbose=0,
                                                       save_best_only=False,
                                                       save_weights_only=True,
                                                       mode='auto', period=1)])


if __name__ == '__main__':
    main(None)
