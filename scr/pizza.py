
#https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3
#https://github.com/keplr-io/quiver
#https://medium.com/datadriveninvestor/building-powerful-image-classification-convolutional-neural-network-using-keras-a1839d0ff298
#https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
#https://github.com/keras-team/keras/blob/master/examples/babi_memnn.py
#https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/
#https://towardsdatascience.com/image-recognition-with-keras-convolutional-neural-networks-e2af10a10114
#https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
#https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
import os

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions
from quiver_engine import server 

import numpy as np
import matplotlib
import cv2

train_data_dir = r'../data/pizza/train'
valid_data_dir = r'../data/pizza/validation'
pizza_w = r'models/simple_pizzaw_CNN.h5'
pizza_m = r'models/simple_pizzam_CNN.h5'

img_width = 150
img_height = 150
classe_names = ['normal','burned']

from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#img = input_img_data[0]
#img = deprocess_image(img)
#imsave('%s_filter_%d.png' % (layer_name, filter_index), img)

def train(epochs=20, img_width=150, img_height=150,
		activation='relu', optimizer='rmsprop',
		dropout=.2):#Dropout rate of 20% to prevent overfitting.
	#dropout rate can be specified to the layer as the probability of setting each input to the layer to zero
	"""
	"""
	print('Starting')
	datagen = ImageDataGenerator(rescale = 1./255,
	# shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
	)
	#test_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = datagen.flow_from_directory(directory=train_data_dir,
						target_size=(img_width,img_height),
						classes=classe_names,
						class_mode='binary',
						batch_size=16)
	validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
						target_size=(img_width,img_height),
						classes=classe_names,
						class_mode='binary',
						batch_size=32)

	# build model
	model = Sequential()
	#32 feature maps with the kernel of (3,3).
	model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))

	#x = BatchNormalization()(x)
	model.add(Activation(activation))
	# edges
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation(activation))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation(activation))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Flatten())
	model.add(Dense(64))

	model.add(Activation(activation))
	model.add(Dropout(dropout))
	##Second  Hidden Laye
	model.add(Dense(units=1))#, activation='sigmoid'))
	model.add(Activation('sigmoid'))
	
	model.compile(optimizer=optimizer, 
				loss='binary_crossentropy',  metrics=['accuracy'])#r'adadelta'
	#classifier.compile( optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
	print('starting training....')
		#heatmap=np.mean(conv_layer_output_value, axis=-1)# Heatmap post processing
	#heatmap = np.maximum(heatmap, 0)
	#heatmap /= np.max(heatmap)
	#plt.matshow(heatmap)
	#plt.show()
	training = model.fit_generator(generator=train_generator,
				steps_per_epoch=2048 // 16, epochs=epochs,
				validation_data=validation_generator, validation_steps=832//16)
	model.save_weights(pizza_w)
	model.save(pizza_m)
	return model

def test(image_path, img_width=150, img_height = 150):
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

def DeepDog(input_tensor=None, input_shape=None, alpha=1, classes=1000):

    input_shape = keras.applications.imagenet_utils._obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(int(32*alpha), (3, 3), strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(32*alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    for _ in range(5):
        x = SeparableConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)

    x = SeparableConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(1024 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    
    x = GlobalAveragePooling2D()(x)
    out = Dense(1, activation='sigmoid')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='deepdog')
    return model
 
def main():
	model = train(epochs=10, dropout=.2)
		#20
	print ("\nBurned: ", len(os.listdir('../data/pizza/test/burned')))
	for img2 in os.listdir('../data/pizza/test/burned'):
		prediction2 = test(os.path.join('../data/pizza/test/burned', img2))
		#print('Predicted:', decode_predictions(rediction2 , top=3)[0])
		if prediction2 is not 'burned':
			print (img2, prediction2, prediction2 is'burned')
		#assert(prediction2 =='burned')
	print ("\nNot burned: ", len(os.listdir('../data/pizza/test/normal')))
	for img3 in os.listdir('../data/pizza/test/normal'):
		prediction3 = test(os.path.join('../data/pizza/test/normal', img3))
		if prediction3 is 'burned':
			print (img3, prediction3, prediction3 is not 'burned')
		print('Predicted:', decode_predictions(prediction3 , top=3)[0])
		#assert(prediction3 is not 'burned')
	#Now is the moment of truth. we check the accuracy on the test dataset
# Grad-CAM algorithm
	specoutput=model.output[:, 668]
	last_conv_layer = model.get_layer('block5_conv3')
	grads = K.gradients(specoutput, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))
	iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
	pooled_grads_value, conv_layer_output_value = iterate([image])
	for i in range(512):
	    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
	heatmap=np.mean(conv_layer_output_value, axis=-1)# Heatmap post processing
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)
	matplotlib.matshow(heatmap)
	matplotlib.show()


	"""
	img = cv2.imread('croppedstockholm4.jpg')
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	superimposed_img = heatmap * 0.4 + img
	cv2.imwrite('heatmap4.jpg', superimposed_img)
	"""

	from quiver_engine import server 
	#server.launch(model, classes=classe_names, input_folder='../data/pizza/train/burned')
	server.launch( model, classe_names,  5, temp_folder='.', input_folder='../data/pizza/train',  port=5000,  mean=[123.568, 124.89, 111.56],  std=[52.85, 48.65, 51.56])

	return

main()
