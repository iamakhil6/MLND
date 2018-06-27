from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras_preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)
# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

import cv2
import matplotlib.pyplot as plt

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))
print('Coordinates of faces ',faces)
# get bounding box for each detected face
for (x, y, w, h) in faces:
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
print(type(human_files_short))
human_count = 0
dog_count = 0
for  i in range(100):
    human_match = face_detector(human_files_short[i])
    if human_match == True:
        human_count += 1
    dog_match = face_detector(dog_files_short[i])
    if dog_match== True:
        dog_count += 1
print('Percentage of first 100 human images on Human Face Classifier ', float(human_count/100)*100)
print('Percentage of first 100 dog images on Human Face Classifier ', float(dog_count/100)*100)


from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
print(ResNet50_predict_labels(human_files_short[0]))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
print(dog_detector(dog_files_short[0]),dog_files_short[0])

h_count =0
d_count = 0
for i in range(len(human_files_short)):
    human_match = dog_detector(human_files_short[i])
    if human_match == True:
        h_count += 1
    dog_match = dog_detector(dog_files_short[i])
    if dog_match == True:
        d_count +=1
print('Percentage of 100 human images on dog detector ', float(h_count/100)*100)
print('Percentage of 100 dog images on dog detector ', float(d_count/100)*100)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
# train_tensors = paths_to_tensor(train_files).astype('float32')/255
# valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
# test_tensors = paths_to_tensor(test_files).astype('float32')/255

# model = Sequential()
# model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224,3)))
# model.add(MaxPooling2D(pool_size=2, padding='same'))
# model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2, padding='same'))
# model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2, padding='same'))
# model.add(GlobalAveragePooling2D(input_shape= (28, 28, 3)))
# model.add(Dense(133, activation='softmax'))
# model.summary()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# epochs = 5
# checkpointer = ModelCheckpoint('saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
# train_datagen = ImageDataGenerator(height_shift_range=0.3, width_shift_range=0.3, vertical_flip=True, horizontal_flip=True, zoom_range=0.2)
# valid_datagen = ImageDataGenerator(height_shift_range=0.3, width_shift_range=0.3, vertical_flip=True, horizontal_flip=True, zoom_range=0.2)
# train_datagen.fit(train_tensors)
# valid_datagen.fit(valid_tensors)
# model.fit_generator(train_datagen.flow(train_tensors, train_targets, batch_size=32), epochs=epochs, validation_data=valid_datagen.flow(valid_tensors, valid_targets, batch_size=32), verbose=1, callbacks=[checkpointer])
# model.load_weights('saved_models/weights.best.from_scratch.hdf5')
# dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
#
# # report test accuracy
# test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)
#
# print('Accuracy ', model.evaluate(test_tensors, test_targets))

bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape= (train_VGG16.shape[1:])))
VGG16_model.add(Dense(133, activation='softmax'))
VGG16_model.summary()
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', verbose=1, save_best_only=True)
VGG16_model.fit(train_VGG16, train_targets, validation_data=(valid_VGG16, valid_targets), epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

VGG16_predictions = [np.argmax(VGG16_model.predict(feature)) for feature in test_VGG16]
test_accuracy= 100*np.sum(np.array(VGG16_predictions)== np.argmax(test_targets, axis=1)/len(VGG16_predictions))
print('Test accuracy: %.4f%%' % test_accuracy)