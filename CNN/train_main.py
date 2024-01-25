import os
from trainutils import *

_size = 256
_z_size = 64
data_dir = '/dev/shm/3dcentroids/' #directory with the folders "train/images"
                                 #"train/outputs", "val/images", "val/outputs"
save_dir = '/mnt/2TBData/hemaxi/centernet/model1' #directory to save
                                                         #the models and the log file

# Parameters
data_train_configs = {'dim': (_size,_size,_z_size,2),
                                        'mask_dim':(_size,_size,_z_size,2),
                                        'batch_size': 2,
                                        'shuffle': True}

data_val_test_configs = {'dim': (_size,_size,_z_size,2),
                                                'mask_dim':(_size,_size,_z_size,2),
                                                'batch_size': 2,
                                                'shuffle': True}

training_configs = {'initial_learning_rate':0.01,
                'learning_rate_drop':0.8,
                'learning_rate_patience':10,
                'learning_rate_epochs':None, 
                'early_stopping_patience':50,
                'n_epochs':100}

# Generators
train_generator = DataGenerator(data_dir, partition='train', configs=data_train_configs, data_aug=True) 
validation_generator = DataGenerator(data_dir, partition='val', configs=data_val_test_configs, data_aug=False)

model = threeDUVec2() #training from scratch
model.summary()

model = train_model(model=model, model_file=os.path.join(save_dir, 'best_centroids.hdf5'), logging_file= os.path.join(save_dir, "centroids.log"),
						training_generator=train_generator,
                        validation_generator=validation_generator,
                        steps_per_epoch=train_generator.__len__(),
                        validation_steps=validation_generator.__len__(), **training_configs)

model.save(os.path.join(save_dir,'final_centroids.hdf5'))
