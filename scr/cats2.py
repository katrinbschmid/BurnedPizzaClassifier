# Convolutional Neural Network (Cats vs Dogs)

# Building the CNN
#https://github.com/hatemZamzam/Cats-vs-Dogs-Classification-CNN-Keras-/blob/master/cnn.py
# Importing the Keras libraries and packages
import os 

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
pizza_w = r'models/simple_1pizzaw_CNN.h5'
pizza_m = r'models/simple_1pizzam_CNN.h5'
def train():
    # Initialising the CNN
    classifier = Sequential()
    
    # Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    
    # Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Flattening
    classifier.add(Flatten())
    
    # Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the CNN to the images
    
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_data_dir = r'../data/pizza/train'
    valid_data_dir = r'../data/pizza/validation'
    training_set = train_datagen.flow_from_directory(train_data_dir,
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory(valid_data_dir,
                                                 target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
    steps = 8000
    steps = 2000
    classifier.fit_generator(training_set,
                             steps_per_epoch = steps,#
                             epochs = 3,
                             validation_data = test_set,
                             validation_steps = 2000)
    ##Prediction Part

    
    model = classifier

    model.save_weights(pizza_w)
    model.save(pizza_m)

#ValueError: Error when checking input: expected conv2d_1_input to have shape (64, 64, 3) but got array with shape (150, 150, 3)
def test(image_path, img_width=64, img_height=64):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # load the model we saved
    model = load_model(pizza_m)
    model.load_weights(pizza_w)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    classes = model.predict_classes(images, batch_size=10)
    #
    if classes[0][0] == 1:
        prediction = 'normal'
    else:
        prediction = 'burned'
    return prediction

print ("\nBurned: ", len(os.listdir('../data/pizza/test/burned')))
for img2 in os.listdir('../data/pizza/test/burned'):
    prediction2 = test(os.path.join('../data/pizza/test/burned', img2))
    if prediction2 is not 'burned':
        print (img2, prediction2, prediction2 is'burned')
    #assert(prediction2 =='burned')
print ("\nNot burned: ", len(os.listdir('../data/pizza/test/normal')))
for img3 in os.listdir('../data/pizza/test/normal'):
    prediction3 = test(os.path.join('../data/pizza/test/normal', img3))
    if prediction3 is 'burned':
        print (img3, prediction3, prediction3 is not 'burned')

img_pred = image.load_img('D:/deeplearning/[FreeTutorials.Us] deeplearning/10 Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_5.jpg', target_size = (64, 64))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = classifier.predict(img_pred)

ind = training_set.class_indices

if rslt[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

##Save model to json
import os
from keras.models import model_from_json

clssf = classifier.to_json()
with open("CatOrDog.json", "w") as json_file:
    json_file.write(clssf)
classifier.save_weights("CorDweights.h5")
print("model saved to disk....")
