from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
# step 1: load data

img_width = 150
img_height = 150
train_data_dir = r'../data/pizza/train'
valid_data_dir = r'../data/pizza/validation'

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['burned','normal'],
											   class_mode='binary',
											   batch_size=16)

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['burned','normal'],
											   class_mode='binary',
											   batch_size=32)


# step-2 : build model

model = Sequential()

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
#model.add(Dropout(0.5))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print('model complied!!')

print('starting training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=2048 // 16,epochs=20,validation_data=validation_generator,validation_steps=832//16)

print('training finished!!')

print('saving weights to simple_CNN.h5')

model.save_weights('models/simple_CNN.h5')

print('all weights saved successfully !!')
model.load_weights('models/simple_CNN.h5')


def test(model, image_path, img_width=150, img_height = 150):
	from keras.preprocessing import image
	img = image.load_img(image_path, target_size=(img_width, img_height))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])

	# load the model we saved
	#model = load_model(pizza_m)
	#model.load_weights(pizza_w)
	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	classes = model.predict_classes(images, batch_size=10)
	if classes[0][0] == 1:
		prediction = 'normal'
	else:
		prediction = 'burned'
	return prediction

print ("\nBurned: ", len(os.listdir('../data/pizza/test/burned')))#4/12 6
for img2 in os.listdir('../data/pizza/test/burned'):
	prediction2 = test(model, os.path.join('../data/pizza/test/burned', img2))
	if prediction2 is not 'burned':
		print (img2, prediction2, prediction2 is'burned')
	#assert(prediction2 =='burned')
print ("\nNot burned: ", len(os.listdir('../data/pizza/test/normal')))#2/8 3
for img3 in os.listdir('../data/pizza/test/normal'):
	prediction3 = test(model, os.path.join('../data/pizza/test/normal', img3))
	if prediction3 is 'burned':
		print (img3, prediction3, prediction3 is not 'burned')
			