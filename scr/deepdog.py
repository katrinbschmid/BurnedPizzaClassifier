
#from keras.applications.imagenet_utils import _obtain_input_shape
import keras.applications.imagenet_utils
from keras import backend as K
from keras.layers import Input, Convolution2D, SeparableConvolution2D, \
                         Activation, BatchNormalization#Dense,GlobalAveragePooling2d, 
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils

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

DeepDog()