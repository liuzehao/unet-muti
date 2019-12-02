from __future__ import print_function
import os
import datetime
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Dropout, \
    BatchNormalization
from keras.optimizers import Adam,Adamax
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.core import Lambda
import cv2

PIXEL = 512    #set your image size
BATCH_SIZE = 16
lr = 0.001
EPOCH = 300
X_CHANNEL = 3  # training data channel
Y_CHANNEL = 1  # test data channel
X_NUM = 1000 # your traning data number
X_val=100
pathX = './images/training/'    #change your file path
pathY = './annotations/training/'    #change your file path
pathX_val='./images/validation/'
pathY_val='./annotations/validation/'
seed = 42
random.seed = seed
np.random.seed = seed
class_num=4#class+1
#modelpath='./model.h5'#模型地址
def get_all_files(bg_path):
    files = []

    for f in os.listdir(bg_path):
        if os.path.isfile(os.path.join(bg_path, f)):
            files.append(os.path.join(bg_path, f))
        else:
            files.extend(get_all_files(os.path.join(bg_path, f)))
    files.sort(key=lambda x: int(x[-8:-4]))#排序从小到大
    return files
#data processing
def generator(pathX, pathY,BATCH_SIZE,NUM):
    while 1:
        X_train_files = get_all_files(pathX)
        Y_train_files = get_all_files(pathY)
        a = (np.arange(1, NUM))
        # print(a)
        # cnt = 0
        X = []
        Y = []
        for i in range(BATCH_SIZE):
            index = np.random.choice(a)
            # print(index)
            # print(X_train_files[index])
            img = cv2.imread(X_train_files[index], 1)
            img=cv2.resize(img,(PIXEL,PIXEL))
            # cv2.imshow("a",img)
            # cv2.waitKey(0)
            # print(np.array(img).shape)
            # print(pathX + str(i+1)+'.png')
            #
            # img = img / 255  # normalization
            img = np.array(img).reshape(PIXEL, PIXEL, X_CHANNEL)
            X.append(img)
            img1 = cv2.imread(Y_train_files[index], 0)
            # print(img1.shape)
            img1=cv2.resize(img1,(PIXEL,PIXEL))
            # img1 = img1 / 255  # normalization
            img1 = np.array(img1).reshape(PIXEL, PIXEL)
            new_label = np.zeros(img1.shape + (class_num,))
            for i in range(class_num):
                new_label[img1 == i, i] = 1
            Y.append(new_label)

            #cnt += 1
            # print(new_label.shape)
            # cv2.imshow("aaa",new_label[:,:,0]*50)
            # cv2.imshow("bbb",new_label[:,:,1]*50)
            # cv2.imshow("ccc",new_label[:,:,2]*50)
            # cv2.imshow("ccc",new_label[:,:,3]*50)
            # cv2.imshow("a",img)
            # cv2.imshow("b",img1*50)
            # cv2.waitKey(0)

        X = np.array(X)
        Y = np.array(Y)
        yield X, Y

def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
inputs = Input((PIXEL, PIXEL, 3))
s = Lambda(lambda x: x / 255) (inputs)
conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(s)
pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  # 16

conv2 = BatchNormalization(momentum=0.99)(pool1)
conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
conv2 = BatchNormalization(momentum=0.99)(conv2)
conv2 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
conv2 = Dropout(0.02)(conv2)
pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  # 8

conv3 = BatchNormalization(momentum=0.99)(pool2)
conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
conv3 = BatchNormalization(momentum=0.99)(conv3)
conv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
conv3 = Dropout(0.02)(conv3)
pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 4

conv4 = BatchNormalization(momentum=0.99)(pool3)
conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
conv4 = BatchNormalization(momentum=0.99)(conv4)
conv4 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
conv4 = Dropout(0.02)(conv4)
pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

conv5 = BatchNormalization(momentum=0.99)(pool4)
conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
conv5 = BatchNormalization(momentum=0.99)(conv5)
conv5 = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
conv5 = Dropout(0.02)(conv5)
pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)
# conv5 = Conv2D(35, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
# drop4 = Dropout(0.02)(conv5)
pool4 = AveragePooling2D(pool_size=(2, 2))(pool3)  # 2
pool5 = AveragePooling2D(pool_size=(2, 2))(pool4)  # 1

conv6 = BatchNormalization(momentum=0.99)(pool5)
conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
up7 = (UpSampling2D(size=(2, 2))(conv7))  # 2
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
merge7 = concatenate([pool4, conv7], axis=3)

conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
up8 = (UpSampling2D(size=(2, 2))(conv8))  # 4
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
merge8 = concatenate([pool3, conv8], axis=3)

conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
up9 = (UpSampling2D(size=(2, 2))(conv9))  # 8
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
merge9 = concatenate([pool2, conv9], axis=3)

conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
up10 = (UpSampling2D(size=(2, 2))(conv10))  # 16
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up10)

conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
up11 = (UpSampling2D(size=(2, 2))(conv11))  # 32
conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up11)

# conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
conv12 = Conv2D(class_num, 1,activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
# outputs = Conv2D(1, 1, activation='relu') (conv12)
model = Model(input=inputs, output=conv12)
# print(model.summary())
model.compile(optimizer=Adamax(lr=1e-5), loss='mse', metrics=['accuracy'])
# earlystopper = EarlyStopping(patience=5, verbose=1)

filepath = "./model.h5"
checkpointer = ModelCheckpoint(filepath,monitor='val_loss',verbose=1, save_best_only=True,period=1)
tensorboard = TensorBoard(log_dir='log',histogram_freq= 0,write_graph=True,write_images=True)
if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")
callback_lists=[tensorboard,checkpointer]
history = model.fit_generator(generator(pathX, pathY,BATCH_SIZE,X_NUM),
                              steps_per_epoch=max(1, X_NUM//BATCH_SIZE),nb_epoch=EPOCH,
                              validation_data=generator(pathX_val,
                              pathY_val,BATCH_SIZE,X_val),validation_steps=max(1, X_val//BATCH_SIZE),callbacks=callback_lists)
end_time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

 #save your training model
model.save(filepath)        

#save your loss data
# mse = np.array((history.history['loss']))
# np.save(r'./V1_828.npy', mse)      
