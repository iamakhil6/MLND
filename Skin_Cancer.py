from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from keras.utils import np_utils

datagen= ImageDataGenerator(zoom_range=0.1, height_shift_range=0.2, width_shift_range=0.2, rescale=1./255)
for x_batch in datagen.flow_from_directory('data/train',batch_size=12, target_size=(32,32)):
    print(len(x_batch[0]))
    fig= plt.figure(figsize=(32,32))
    for i in range(12):
        fig.add_subplot(3,4,1+i)
        plt.imshow(x_batch[0][i])
    plt.show()
    break
train_data= datagen.flow_from_directory('data/train', batch_size=100, target_size=(32,32))
valid_data= datagen.flow_from_directory('data/valid', batch_size=100, target_size=(32,32))
# test_data= datagen.flow_from_directory('data/test', target_size=(32,32))
def load_data(path):
    data= load_files(path)
    features= np.array(data['filenames'])
    labels= np_utils.to_categorical(data['target'], 3)
    return features, labels
x_test, y_test= load_data('data/test')
model= Sequential()
model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding='same',input_shape=(32,32,3)))
# model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
# model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss= 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# best_model= ModelCheckpoint('best_model/skin_cancer.hdf5', save_best_only=True, verbose=1)
# stop= EarlyStopping(monitor='val_loss', patience=2)
model.fit_generator(train_data, epochs=1, validation_data= valid_data, shuffle= True,callbacks= 'skin_cancer.hdf5')
model.load_weights('skin_cancer.hdf5')
def convert(path):
    img = load_img(path= path,target_size=(32, 32))
    x= img_to_array(img)
    x= np.expand_dims(x, axis=0)
    return np.argmax(model.predict(x))
predictions=[]
for feature in x_test:
    predictions.append(convert(feature))
accuracy= 100* np.sum(np.array(predictions)== np.argmax(y_test, axis=1)/len(predictions))
print('Accuracy ',accuracy)
