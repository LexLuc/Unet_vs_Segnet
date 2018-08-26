# segnet.py
# An implementation of SegNet refered from https://arxiv.org/abs/1511.00561
# based on keras with tensorflow backend.
#
# Assignment work for COMP 9417, Semester 1, 2018
# Team: Auto_segmentation

import os 
import numpy as np
import keras.backend
import matplotlib.pyplot as plt
from json import dump, load
from keras.utils.np_utils import to_categorical  
from sklearn.preprocessing import LabelEncoder 
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Reshape, Permute, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint


class Segnet(object):
    """
    OO implementation of SegNet for Kaggle image segmentation challange
    on https://www.kaggle.com/c/carvana-image-masking-challenge.
    Features:
    1. Train network on training dataset where the images and masks are seperated in different directories;
    2. Predict masks on test dataset where the images and masks are seperated in different directories;
    3. Plot line chart of loss and accuracy in terms of number of epochs for training set and validation set respectively;
    4. Save the model file and training history for future use.
    """

    def __init__(self, img_rows=256, img_cols=256, batch_size=8, classes=[0, 1]):
        self.img_scale = (img_rows, img_cols, 3)
        self.classes = classes
        self.n_label = len(classes)
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(self.classes)
        self.batch_size = batch_size
        self.train_size = None
        self.valid_size = None
        self.test_size = None


    def _generateData(self, image_dir, label_dir=None):
        imgnames = sorted(os.listdir(image_dir))
        labelnames = [] if label_dir is None else sorted(os.listdir(label_dir))
        
        while True:  
            #random_indices = np.random.choice(np.arange(len(imgnames)), self.batch_size)
            for start in range(0, len(imgnames), self.batch_size):
                data = []  
                label = []
                end = min(start+self.batch_size, len(imgnames))
                #for i in random_indices:
                for i in range(start, end): 
                    image_path = os.path.join(image_dir, imgnames[i])
                    image = load_img(image_path, target_size=self.img_scale)
                    image_arr = img_to_array(image) /255
                    data.append(image_arr)
                    
                    if label_dir is not None:
                        label_path = os.path.join(label_dir, labelnames[i])
                        mask = load_img(label_path, target_size=self.img_scale, grayscale=True)
                        mask_arr = img_to_array(mask).reshape((self.img_scale[0] * self.img_scale[1],))/255
                        label.append(mask_arr)
                    
                data = np.array(data)
                if len(label) == 0:
                    yield data
                else:
                    label = np.array(label).flatten()
                    label = self.labelencoder.transform(label)
                    label = to_categorical(label, num_classes=self.n_label)
                    label = label.reshape((self.batch_size, self.img_scale[0] * self.img_scale[1], self.n_label))  
                    yield data, label


    def load_data(self, train_image_dir, train_label_dir, valid_image_dir, valid_label_dir, test_image_dir):
        self.train_size = len(os.listdir(train_image_dir))
        self.valid_size = len(os.listdir(valid_image_dir))
        self.test_size = len(os.listdir(test_image_dir))
        
        train_generator = self._generateData(train_image_dir, train_label_dir)
        valid_generator = self._generateData(valid_image_dir, valid_label_dir)
        test_generator = self._generateData(test_image_dir)
        
        return train_generator, valid_generator, test_generator


    def _encode(self, encoder, kernel=3):
        encoder.add(Convolution2D(64, (kernel, kernel), input_shape=self.img_scale, padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(64, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(MaxPooling2D())

        encoder.add(Convolution2D(128, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(128, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(MaxPooling2D())

        encoder.add(Convolution2D(256, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(256, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(256, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(MaxPooling2D())

        encoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(MaxPooling2D())

        encoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        encoder.add(MaxPooling2D())

        return encoder


    def _decoder(self, decoder, kernel=3):
    
        decoder.add(UpSampling2D())
        decoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))

        decoder.add(UpSampling2D())
        decoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(512, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(256, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))

        decoder.add(UpSampling2D())
        decoder.add(Convolution2D(256, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(256, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(128, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))

        decoder.add(UpSampling2D())
        decoder.add(Convolution2D(128, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(64, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))

        decoder.add(UpSampling2D())
        decoder.add(Convolution2D(64, (kernel, kernel), padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))
        decoder.add(Convolution2D(self.n_label, (1, 1), padding='valid', activation="sigmoid"))
        decoder.add(BatchNormalization())

        return decoder


    def dice_coef(self, y_true, y_pred):
        smooth = 1e-4
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
        
        return score


    def model(self):

        # refered from https://www.kaggle.com/rdebbe/is-segnet-a-good-model-for-sharp-edge-masking
        model = Sequential()  
        encoder = self._encode(model)
        decoder = self._decoder(encoder)
        
        model.add(Reshape((2, self.img_scale[0]*self.img_scale[1])))  
        model.add(Permute((2,1)))  
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=[self.dice_coef])  
        model.summary()  
        
        return model  


    def train(self, train_generator, valid_generator, n_epoch=20, train_steps=8, save_history=False):

        model = self.model()
        model_checkpoint = ModelCheckpoint('Segnet.hdf5', monitor='loss',verbose=1, save_best_only=True)
        #train_info = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=n_epoch, verbose=1, validation_data=valid_generator, validation_steps=2, callbacks=[model_checkpoint])
        train_info = model.fit_generator(train_generator, steps_per_epoch=self.train_size//self.batch_size, epochs=n_epoch, verbose=1, validation_data=valid_generator, validation_steps=self.valid_size//self.batch_size, callbacks=[model_checkpoint])

        if save_history is True:
            with open('train_history_segnet.json', mode='w', encoding='utf-8') as json_file:
                dump(train_info.history, json_file)


    def plot_history(self, n_epoch=10):
        with open('train_history_segnet.json', mode='r', encoding='utf-8') as json_file:
            history = load(json_file)
        
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, n_epoch), history['loss'], label='train loss')
        plt.plot(np.arange(0, n_epoch), history['val_loss'], label='val_loss')
        plt.plot(np.arange(0, n_epoch), history['dice_coef'], label='train_dice_coef')
        plt.plot(np.arange(0, n_epoch), history['val_dice_coef'], label='val_dice_coef')
        plt.title('Training performance of SegNet')
        plt.xlabel('#Epochs')
        plt.ylabel('Loss/Dice_coef')
        plt.legend(loc='best')
        plt.savefig('loss_dicecoef_segnet.png')


    def predict(self, weight_dir, test_generator, test_label_dir):
        model = load_model(weight_dir, custom_objects=dict(dice_coef=self.dice_coef))
        
        for step in range(2):
            test_images = next(test_generator)
            arr_predicted = model.predict_classes(test_images, verbose=2)
            assert test_images.shape[1:] == self.img_scale
            
            for i in range(arr_predicted.shape[0]):
                print('*'*10, i*step, '*'*10)
                arr_predicted_2d = arr_predicted[i].reshape(self.img_scale[0], self.img_scale[1])
                arr_predicted_2d = arr_predicted_2d.astype(np.uint8)
                arr_predicted_3d = np.stack((arr_predicted_2d,)*3, -1)
                mask_predicted = array_to_img(arr_predicted_3d)
                test_label_path = os.path.join(test_label_dir, '{}_segnet.gif'.format(i*step))
                mask_predicted.save(test_label_path)


if __name__ == '__main__':

    data_dir = os.path.join('.', 'data')
    train_image_dir = os.path.join(data_dir, 'train', 'image', 'car')
    train_label_dir= os.path.join(data_dir, 'train', 'label', 'car')
    valid_image_dir = os.path.join(data_dir, 'train', 'validation_image', 'car')
    valid_label_dir = os.path.join(data_dir, 'train', 'validation_label', 'car')
    test_image_dir = os.path.join(data_dir, 'test', 'img', 'car')
    test_label_dir = os.path.join(data_dir, 'test', 'label', 'car')
    
    n_epoch = 15
    
    segnet = Segnet()
    train_gen, valid_gen, test_gen = segnet.load_data(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir, test_image_dir)
    segnet.train(train_gen, valid_gen, n_epoch=n_epoch, save_history=True)
    segnet.plot_history(n_epoch=n_epoch)
    segnet.predict('Segnet.hdf5', test_gen, test_label_dir)

