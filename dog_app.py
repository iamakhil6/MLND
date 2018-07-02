import keras
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
print(type(human_files_short), human_files_short[0])
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
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

# ----------------------------------   From Scratch    ------------------------------------------------------------

model = Sequential()
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224,3)))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(GlobalAveragePooling2D(input_shape= (28, 28, 3)))
model.add(Dense(133, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10

checkpointer = ModelCheckpoint('saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
train_datagen = ImageDataGenerator(height_shift_range=0.3, width_shift_range=0.3, vertical_flip=True, horizontal_flip=True, zoom_range=0.2)
valid_datagen = ImageDataGenerator(height_shift_range=0.3, width_shift_range=0.3, vertical_flip=True, horizontal_flip=True, zoom_range=0.2)
train_datagen.fit(train_tensors)
valid_datagen.fit(valid_tensors)
model.fit_generator(train_datagen.flow(train_tensors, train_targets, batch_size=32), epochs=epochs, validation_data=valid_datagen.flow(valid_tensors, valid_targets, batch_size=32), verbose=1, callbacks=[checkpointer])
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# -------------------------------  VGG16 Model  ------------------------------------------------------------------

# bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
# train_VGG16 = bottleneck_features['train']
# valid_VGG16 = bottleneck_features['valid']
# test_VGG16 = bottleneck_features['test']
#
# VGG16_model = Sequential()
# VGG16_model.add(GlobalAveragePooling2D(input_shape= (train_VGG16.shape[1:])))
# VGG16_model.add(Dense(133, activation='softmax'))
# VGG16_model.summary()
# VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', verbose=1, save_best_only=True)
# VGG16_model.fit(train_VGG16, train_targets, validation_data=(valid_VGG16, valid_targets), epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)
#
# VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
#
# VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]
# test_accuracy= 100*np.sum(np.array(VGG16_predictions)== np.argmax(test_targets, axis=1))/len(VGG16_predictions)

#--------------------------------  VGG19 Model  ---------------------------------------------------------------------

# bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
# train_VGG19 = bottleneck_features['train']
# test_VGG19 = bottleneck_features['test']
# valid_VGG19 = bottleneck_features['valid']
#
# VGG19_model= Sequential()
# print('Shape ',train_VGG19.shape[1:])
# VGG19_model.add(GlobalAveragePooling2D(input_shape= train_VGG19.shape[1:]))
# VGG19_model.add(Dense(133, activation='softmax'))
# VGG19_model.summary()
# VGG19_model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='rmsprop')
# checkpointer = ModelCheckpoint('saved_models/weights.best.VGG19.hdf5', save_best_only=True, verbose=1)
# train_datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, width_shift_range=0.3, height_shift_range=0.2,rotation_range=0.2, data_format='channels_last')
# train_datagen.fit(train_VGG19)
# valid_datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, width_shift_range=0.3, height_shift_range=0.2,rotation_range=0.2, data_format='channels_last')
# valid_datagen.fit(valid_VGG19)
# # DogVGG19_model.fit(train_VGG19, train_targets, epochs= 10, batch_size=20, validation_data=(valid_VGG19, valid_targets), verbose=1, callbacks=[checkpointer])
# VGG19_model.fit_generator(train_datagen.flow(train_VGG19, train_targets, 20), epochs= 70, validation_data= valid_datagen.flow(valid_VGG19, valid_targets, 20), verbose=1, callbacks=[checkpointer])
# VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')
# VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]
# test_accuracy = 100 * np.sum(np.array(VGG19_predictions) == np.argmax(test_targets, axis=1))/len(VGG19_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)

# -------------------------------------------  InceptionV3  ------------------------------------------------------------------

'''
Accuracy improved because we used pre trained model along with Image Augmentation on our dataset
This model is successful for image classification compared to VGG16, VGG19, RESNET because of it's architecture
'''
epochs = 5
# Instantiate Pre trained model into bottleneck_features
bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
test_InceptionV3 = bottleneck_features['test']
valid_InceptionV3 = bottleneck_features['valid']

InceptionV3_model= Sequential()
print('Shape ',train_InceptionV3.shape[1:])

# We are connecting last layer of InceptionV3 model to GAP layer and dense layer where one node is allocated for each dog category for the later
InceptionV3_model.add(GlobalAveragePooling2D(input_shape= train_InceptionV3.shape[1:]))
InceptionV3_model.add(Dense(133, activation='softmax'))
InceptionV3_model.summary()
InceptionV3_model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='rmsprop')

# To save best model weights which we will use for predictions later
checkpointer = ModelCheckpoint('saved_models/weights.best.DogInceptionV3Data.hdf5', save_best_only=True, verbose=1)
train_datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, width_shift_range=0.3, height_shift_range= 0.2, rotation_range=0.2, data_format='channels_last')
train_datagen.fit(train_InceptionV3)

# Generate batches of tensor image data with real-time data augmentation. So that same image won't repeat more than twice will help in improving efficiency
valid_datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, width_shift_range=0.3, height_shift_range= 0.2, rotation_range=0.2, data_format='channels_last')
valid_datagen.fit(valid_InceptionV3)

# Training InceptionV3 model with data generators to achieve better accuracy
InceptionV3_model.fit_generator(train_datagen.flow(train_InceptionV3, train_targets, 30), epochs= epochs, validation_data= valid_datagen.flow(valid_InceptionV3, valid_targets, 20), verbose=1, callbacks=[checkpointer])
InceptionV3_model.load_weights('saved_models/weights.best.DogInceptionV3Data.hdf5')

#  Get index of predicted dog breed in test set
InceptionV3_predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

#  Report test accuracy
test_accuracy = 100 * np.sum(np.array(InceptionV3_predictions) == np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

from extract_bottleneck_features import *

# Method that accepts a file path to an image and determines whether the image contains a human, dog, or neither. And return the predicted breed
def InceptionV3_predict_breed(img_path):
    
    #  To display Image
    tensor = path_to_tensor(img_path)
    image = np.squeeze(tensor, axis=0)
    image = image.astype('float32') / 255
    plt.imshow(image)
    plt.show()
    
    # Another way To display Image
    # z = keras.preprocessing.image.load_img(img_path)
    # z = keras.preprocessing.image.img_to_array(z)
    # z = z.astype('float32') / 255
    # plt.imshow(z)
    # plt.show()
    # extract bottleneck features
    img = extract_InceptionV3(tensor)
    x = 0
    if dog_detector(img_path):
        print('Hello Dog ')
    elif face_detector(img_path):
        print('Hello Human ')
    else:
        x = 1
    if(x == 1):
        return "Please provide valid input image "
    else:
        # obtain predicted vector
        InceptionV3_model.load_weights('saved_models/weights.best.DogInceptionV3Data.hdf5')
        predicted_vector = InceptionV3_model.predict(img)
        
        # return dog breed that is predicted by the model
        return 'You look like a ' + dog_names[np.argmax(predicted_vector)]

def InceptionV3_predict_breed(img_path):
    #  To display Image
    z = keras.preprocessing.image.load_img(img_path)
    z = keras.preprocessing.image.img_to_array(z)
    z = z.astype('float32') / 255
    plt.imshow(z)
    plt.show()
    # extract bottleneck features
    # img = extract_InceptionV3(tensor)
    # x = 0
    if dog_detector(img_path):
        plt.title('Hello Dog! ')
        predicted_vector = InceptionV3_model.predict(img)
        plt.show()
        return 'Your predicted breed is ' + dog_names[np.argmax(predicted_vector)]
    elif face_detector(img_path):
        plt.title('Hello Human ')
        plt.show()
        predicted_vector = InceptionV3_model.predict(img)
        return 'You look like a ' + dog_names[np.argmax(predicted_vector)]
    else:
        plt.show()
        return "Please provide valid input image "
    
print(InceptionV3_predict_breed(dog_files_short[1]))
print(InceptionV3_predict_breed(dog_files_short[20]))
print(InceptionV3_predict_breed(dog_files_short[10]))
print(InceptionV3_predict_breed(dog_files_short[0]))
print(InceptionV3_predict_breed(human_files_short[0]))
print(InceptionV3_predict_breed(human_files_short[10]))
print(InceptionV3_predict_breed(human_files_short[21]))
