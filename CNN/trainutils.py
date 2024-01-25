from keras import backend as K
from tensorflow.python.ops import *
import tensorflow as tf
import math
from functools import partial
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU
from keras.optimizers import Adam
K.set_image_data_format("channels_last")
try:
        from keras.engine import merge
except ImportError:
        from keras.layers.merge import concatenate
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras 
from keras.losses import categorical_crossentropy
from keras import layers as L
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from skimage import data
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import keras
from batchgenerators.augmentations.spatial_transformations import *

import cv2
from scipy.ndimage.interpolation import rotate
import random

def mse(y_true, y_pred, sample_weight=None):
    squared  = math_ops.square(y_pred - y_true)
    if sample_weight==None:
        return tf.reduce_mean(squared)
    else:
        multiplication = math_ops.multiply(sample_weight, squared)
        return tf.reduce_mean(multiplication)

def mean_se(y_true, y_pred):
    [vecxgt, vecygt, weightx, weighty] = tf.unstack(y_true, 4, axis=4)
    [vecx, vecy] = tf.unstack(y_pred, 2, axis=4)
    mse_loss_channelx = mse(vecx, vecxgt)
    mse_loss_channely = mse(vecy, vecygt)
    return (1/2)*mse_loss_channelx + (1/2)*mse_loss_channely 

def weighted_mean_se(y_true, y_pred):
    [vecxgt, vecygt, weightx, weighty] = tf.unstack(y_true, 4, axis=4)
    [vecx, vecy] = tf.unstack(y_pred, 2, axis=4)
    mse_loss_channelx = mse(vecx, vecxgt, weightx)
    mse_loss_channely = mse(vecy, vecygt, weighty)
    return 0.5*mse_loss_channelx + 0.5*mse_loss_channely 

def threeDUVec2(n_classes=2, im_sz=256, depth=64, n_channels=2, n_filters_start=8, growth_factor=2, upconv=True):
        droprate=0.10
        n_filters = n_filters_start
        inputs = Input((im_sz, im_sz, depth, n_channels))
        #inputs = BatchNormalization(axis=-1)(inputs)
        conv1 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(inputs)
        conv1 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv1)
        #pool1 = Dropout(droprate)(pool1)

        n_filters *= growth_factor
        pool1 = BatchNormalization(axis=-1)(pool1)
        conv2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
        conv2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv2)
        pool2 = Dropout(droprate)(pool2)

        n_filters *= growth_factor
        pool2 = BatchNormalization(axis=-1)(pool2)
        conv3 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
        conv3 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv3)
        pool3 = Dropout(droprate)(pool3)

        n_filters *= growth_factor
        pool3 = BatchNormalization(axis=-1)(pool3)
        conv4_0 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
        conv4_0 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4_0)
        pool4_1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv4_0)
        pool4_1 = Dropout(droprate)(pool4_1)

        n_filters *= growth_factor
        pool4_1 = BatchNormalization(axis=-1)(pool4_1)
        conv4_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool4_1)
        conv4_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4_1)
        pool4_2 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv4_1)
        pool4_2 = Dropout(droprate)(pool4_2)

        n_filters *= growth_factor
        conv5 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool4_2)
        conv5 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)

        n_filters //= growth_factor
        if upconv:
                up6_1 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv5), conv4_1])
        else:
                up6_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4_1])
        up6_1 = BatchNormalization(axis=-1)(up6_1)
        conv6_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up6_1)
        conv6_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6_1)
        conv6_1 = Dropout(droprate)(conv6_1)

        n_filters //= growth_factor
        if upconv:
                up6_2 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv6_1), conv4_0])
        else:
                up6_2 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_1), conv4_0])
        up6_2 = BatchNormalization(axis=-1)(up6_2)
        conv6_2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up6_2)
        conv6_2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6_2)
        conv6_2 = Dropout(droprate)(conv6_2)

        n_filters //= growth_factor
        if upconv:
                up7 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv6_2), conv3])
        else:
                up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv3])
        up7 = BatchNormalization(axis=-1)(up7)
        conv7 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up7)
        conv7 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)
        conv7 = Dropout(droprate)(conv7)

        n_filters //= growth_factor
        if upconv:
                up8 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv7), conv2])
        else:
                up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2])
        up8 = BatchNormalization(axis=-1)(up8)
        conv8 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up8)
        conv8 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv8)
        conv8 = Dropout(droprate)(conv8)

        n_filters //= growth_factor
        if upconv:
                up9 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv8), conv1])
        else:
                up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1])
        conv9 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(up9)
        conv9 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(conv9)

        conv11 = Conv3D(n_classes, (1, 1, 1), activation='sigmoid', data_format='channels_last')(conv9)

        model = Model(inputs=inputs, outputs=conv11)    
        model.compile(optimizer=Adam(), loss=weighted_mean_se, metrics = [mean_se])
        return model

#Learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
        return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, logging_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                                    learning_rate_patience=50, verbosity=1,
                                    early_stopping_patience=None):
        callbacks = list()
        callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
        callbacks.append(CSVLogger(logging_file, append=True))
        if learning_rate_epochs:
                callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                                                                             drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
        else:
                callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                                                                     verbose=verbosity))
        if early_stopping_patience:
                callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
        return callbacks


def load_old_model(model_file):
        print("Loading pre-trained model")
        custom_objects = {'mean_se': mean_se, 'mse':mse, 'weighted_mean_se':weighted_mean_se}
        try:
                from keras_contrib.layers import InstanceNormalization
                custom_objects["InstanceNormalization"] = InstanceNormalization
        except ImportError:
                pass
        try:
                return load_model(model_file,custom_objects=custom_objects)
        except ValueError as error:
                if 'InstanceNormalization' in str(error):
                        raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                                                                    "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
                else:
                        raise error


def train_model(model, model_file, logging_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                                learning_rate_patience=20, early_stopping_patience=20):
        model.fit_generator(generator=training_generator,
                                                steps_per_epoch=steps_per_epoch,
                                                epochs=n_epochs,
                                                validation_data=validation_generator,
                                                validation_steps=validation_steps,
                                                callbacks=get_callbacks(model_file, logging_file,
                                                                                                initial_learning_rate=initial_learning_rate,
                                                                                                learning_rate_drop=learning_rate_drop,
                                                                                                learning_rate_epochs=learning_rate_epochs,
                                                                                                learning_rate_patience=learning_rate_patience,
                                                                                                early_stopping_patience=early_stopping_patience))
        return model 

# Generates data for Keras
class DataGenerator(keras.utils.Sequence):

        def __init__(self, data_dir, partition, configs, data_aug):

                self.data_aug = data_aug
                self.partition = partition
                self.data_dir = data_dir
                self.list_IDs = sorted(os.listdir(self.data_dir+partition+'/images/'),key=self.order_dirs)

                self.dim = configs['dim']
                self.mask_dim = configs['mask_dim']
                self.batch_size = configs['batch_size']
                self.shuffle = configs['shuffle']
                self.on_epoch_end()

        def __len__(self):
                'Denotes the number of batches per epoch'
                return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):
                'Generate one batch of data'
                # Generate indexes of the batch
                indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
                # Find list of IDs
                list_IDs_temp = [self.list_IDs[k] for k in indexes]
                # Generate data
                X, mask = self.__data_generation(list_IDs_temp)
                X, mask = self.norm_(X,mask)
                X, mask = self.compute_weights(X,mask)    
                return X, mask

        def on_epoch_end(self):
                'Updates indexes after each epoch'
                self.indexes = np.arange(len(self.list_IDs))
                if self.shuffle == True:
                        np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
                # Initialization
                X = np.empty((self.batch_size, *self.dim))
                mask = np.empty((self.batch_size, *self.mask_dim))
                # Generate data
                for i, ID_path in enumerate(list_IDs_temp):

                        img_aux = np.load(self.data_dir + self.partition +'/images/'+ ID_path)
                        msk_aux = np.load(self.data_dir + self.partition +'/outputs/'+ ID_path)


                        if self.data_aug:
                            img_aux = img_aux/255.0
                            msk_aux = msk_aux/255.0
                            if(random.uniform(0,1)<0.5):
                                img_aux, msk_aux = self.vertical_flip(img_aux, msk_aux)
                            if(random.uniform(0,1)<0.5):
                                img_aux, msk_aux = self.horizontal_flip(img_aux, msk_aux)
                            if(random.uniform(0,1)<0.5):
                                img_aux, msk_aux = self.intensity(img_aux, msk_aux)
                            if(random.uniform(0,1)<0.5):
                                angle = np.random.choice(np.arange(0,360,90))
                                img_aux, msk_aux = self.rotation(img_aux, msk_aux, angle)

                            img_aux = img_aux*255.0
                            msk_aux = msk_aux*255.0

                        img_aux = img_aux[:,:,:,0:2]

                        X[i,] = img_aux
                        mask[i,] = msk_aux
                return X, mask


        def compute_weights(self,X,mask):

            mask_out = np.zeros((np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2], np.shape(mask)[3], 4))

            for i in range(0, np.shape(mask)[0]): 
                aux_x = np.zeros((np.shape(mask)[1], np.shape(mask)[2], np.shape(mask)[3]))
                aux_y = np.zeros((np.shape(mask)[1], np.shape(mask)[2], np.shape(mask)[3]))
                num_ = np.shape(mask)[1]*np.shape(mask)[2]*np.shape(mask)[3]
                aux_x[mask[i,:,:,:,0]!=0] = 1000
                aux_x[mask[i,:,:,:,0]==0] = 10
                aux_y[mask[i,:,:,:,1]!=0] = 1000
                aux_y[mask[i,:,:,:,1]==0] = 10
                mask_out[i,:,:,:,0] = mask[i,:,:,:,0]
                mask_out[i,:,:,:,1] = mask[i,:,:,:,1]               
                mask_out[i,:,:,:,2] = aux_x
                mask_out[i,:,:,:,3] = aux_y

            return X, mask_out


        def order_dirs(self, element):
                return element.replace('.npy','')

        def norm_(self, X, mask):
                X, mask = self.rescale_img_values(X, mask)
                return X, mask

        def rescale_img_values(self,img, mask, max=None, min=None):
                img = img/255.0
                mask = mask/255.0
                return img, mask

        ##auxiliary function to rotate the vector
        def vector_rotation(self, x,y, angle):
            angle_rad = (np.pi*angle)/(180)
            rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
            point = np.array([y,x]).T
            [ry, rx] = rot_matrix.dot(point)
            return rx,ry

        ##auxiliary function to rotate the point
        def rotate_around_point_lowperf(self, image, pointx, pointy, angle):
            radians = (np.pi*angle)/(180)
            x, y = pointx, pointy
            ox, oy = image.shape[0]/2, image.shape[1]/2
            qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
            qy = oy - math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)
            return qx, qy

        ##rotation
        def rotation(self, image, mask, angle):
            rot_image = np.zeros(np.shape(image))
            rot_mask = np.zeros(np.shape(mask))
            for z in range(0, rot_image.shape[2]):
                rot_image[:,:,z,:] = rotate(image[:,:,z,:], angle, mode='constant', reshape=False)
                rot_mask[:,:,z,:] = rotate(mask[:,:,z,:], angle, mode='constant', reshape=False)
            return rot_image, rot_mask

        ##vertical flip
        def vertical_flip(self, image, mask):
            flippedimage = np.zeros(np.shape(image))
            flippedmask = np.zeros(np.shape(mask))
            for z in range(0, flippedimage.shape[2]):
                flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 0)
                flippedmask[:,:,z,:] = cv2.flip(mask[:,:,z,:], 0)
            return flippedimage, flippedmask

        ##horizontal flip
        def horizontal_flip(self, image, mask):
            flippedimage = np.zeros(np.shape(image))
            flippedmask = np.zeros(np.shape(mask))
            for z in range(0, flippedimage.shape[2]):
                flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 1)
                flippedmask[:,:,z,:] = cv2.flip(mask[:,:,z,:], 1)
            return flippedimage, flippedmask

        #intensity variations
        def intensity(self, image, mask, alpha=None):
            image = image.astype('float64')
            image = image*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
            image = image.astype('float64')
            return image, mask