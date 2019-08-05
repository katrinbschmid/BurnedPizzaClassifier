
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import numpy as np

train_data_dir = r'../data/pizza/train'
valid_data_dir = r'../data/pizza/validation'
pizza_w = r'models/simple_pizzaw_CNN.h5'
pizza_m = r'models/simple_pizzam_CNN.h5'
		
img_width = 150
img_height = 150
classe_names = ['normal','burned']

def train(epochs=20, img_width=150, img_height=150):
	"""
	"""
	print('Starting')
	datagen = ImageDataGenerator(rescale = 1./255)
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
	model =Sequential()
	model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	
	model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
	
	print('starting training....')# 
	training = model.fit_generator(generator=train_generator, 
				steps_per_epoch=2048 // 16,epochs=epochs,
				validation_data=validation_generator,validation_steps=832//16)
	model.save_weights(pizza_w)
	model.save('models/simple_pizzam_CNN.h5')
	return model

def test(image_path, img_width=150, img_height = 150):
	img = image.load_img(image_path, target_size=(img_width, img_height))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])

	# load the model we saved
	model = load_model('models/simple_pizzam_CNN.h5')
	model.load_weights(pizza_w)
	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	
	classes = model.predict_classes(images, batch_size=10)
	if classes[0][0] == 1:
		prediction = 'normal'
	else:
		prediction = 'burned'
	return prediction

def main():
	train(epochs=20)
	img0 = r'../data/test/5cd5c7261f000030009e6a98.jpeg'
	img1 = r'../data/test/o.jpg'
	img2 = r'../data/test/01-most-instagram-worthy-restaurants.jpg'
	img3 = r'../data/test/pizza_amelia_hitchens.jpg'
	prediction0 = test(img0)
	print (img0, prediction0)
	prediction1 = test(img1)
	print (img1, prediction1)
	prediction2 = test(img2)
	print (img2, prediction2)
	prediction3 = test(img3)
	print (img3, prediction3)
	return

main()
