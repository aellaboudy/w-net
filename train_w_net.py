from w_net_v11 import get_unet
from data_loader import get_data_generators
from keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam, Adadelta
import horovod.keras as hvd
from keras import backend as K    


def main(args):
    img_rows = 128
    img_cols = 832/2
    batch_size = 1
    n_epochs = 100
    models_folder = 'models'
    model_name = 'w_net_V12'
    model_path = os.path.join(models_folder, model_name)

    # Initialize Horovod
    print ('Initializing')
    hvd.init()
    print ('Initialized')

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    # Adjust learning rate based on number of GPUs.
    opt = Adadelta()

    # Add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)
    
    train_generator, val_generator, training_samples, val_samples = get_data_generators(train_folder='/home/ubuntu/data/stereoimages/images/train/',
                                                                                        val_folder='/home/ubuntu/data/stereoimages/images/val/',
                                                                                        img_rows=img_rows,
                                                                                        img_cols=img_cols,
                                                                                        batch_size=batch_size)

    print('found {} training samples and {} validation samples'.format(training_samples, val_samples))
    print('...')
    print('building model...')

    w_net, disp_map_model = get_unet(img_rows=img_rows, img_cols=img_cols, lr=1e-7)


    w_net.compile(optimizer=opt, loss='mean_absolute_error', loss_weights=[1.,0.,0.001,0.001])
    
    callbacks = [
    	# Broadcast initial variable states from rank 0 to all other processes.
    	# This is necessary to ensure consistent initialization of all workers when
    	# training is started with random weights or restored from a checkpoint.
    	hvd.callbacks.BroadcastGlobalVariablesCallback(0),

 	ReduceLROnPlateau(patience=3, verbose=1)
    ]

    model_path = os.path.join(models_folder, model_name)
    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
    	callbacks.append(ModelCheckpoint(model_path + '.h5', monitor='loss',
                                                       verbose=0,
                                                       save_best_only=False,
                                                       save_weights_only=True,
                                                       mode='auto', period=1))
	
    #w_net.load_weights(model_path + '.h5')
    #print('saving model to {}...'.format(model_path))
    #model_yaml = w_net.to_yaml()
    #with open(model_path + ".yaml", "w") as yaml_file:


    print('begin training model, {} epochs...'.format(n_epochs))
    print('Validation steps {} \n'.format(val_samples//batch_size))
    model_path = os.path.join(models_folder, model_name)

    
    print('begin training model, {} epochs...'.format(n_epochs))


    w_net.fit_generator(train_generator,
                            steps_per_epoch=(training_samples // batch_size)//hvd.size(),
                            epochs=n_epochs,
                            validation_data=val_generator,
                            validation_steps=(val_samples // batch_size)//hvd.size(),
                            verbose=1, callbacks=callbacks)



if __name__ == '__main__':
    main(None)
