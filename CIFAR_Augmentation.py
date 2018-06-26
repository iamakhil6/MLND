from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test)= cifar10.load_data()
for i in range(12):
    plt.subplot(3,4,1+i)
    plt.imshow(x_train[i])
plt.show()

x_train= x_train.astype('float32')/255
x_test= x_test.astype('float32')/255
x_valid= x_train[:5000]
y_valid= y_train[:5000]
x_train= x_train[5000:]
y_train= y_train[5000:]
num_classes= len(np.unique(y_train))
y_train=  np_utils.to_categorical(y_train, num_classes)
y_valid= np_utils.to_categorical(y_valid, num_classes)
y_test= np_utils.to_categorical(y_test, num_classes)
print('No of Unique classes ', num_classes)
print('Total sample in train',x_train.shape[0])
print('Total sample in valid',x_valid.shape[0])
print('Total sample in test',x_test.shape[0])
datagen_train= ImageDataGenerator(horizontal_flip= True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1)
datagen_train.fit(x_train)
datagen_valid= ImageDataGenerator(horizontal_flip= True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1)
datagen_valid.fit(x_valid)
for x_batch in datagen_train.flow(x_train, batch_size=12):
    print(type(x_batch), len(x_batch))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(x_batch[i])
    plt.show()
    break
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D
model= Sequential()
model.add(Conv2D(filters= 16, kernel_size=3,  input_shape=(32,32,3), activation='relu', padding= 'same'))
model.add(Conv2D(filters= 16, kernel_size=3, activation='relu', padding= 'same'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64, padding='same',kernel_size=3, activation='relu'))
model.add(Conv2D(filters=64, padding='same', kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
epochs= 70
batch_size= 100
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
checkpoint= ModelCheckpoint('cifar_best.hdf5', save_best_only= True, verbose=1)
model.fit_generator(datagen_train.flow(x_train, y_train, batch_size),callbacks=[checkpoint], epochs=epochs, validation_data=datagen_valid.flow(x_valid, y_valid, batch_size=batch_size))
model.load_weights('cifar_best.hdf5')
print('Accuracy ',model.evaluate(x_test, y_test))