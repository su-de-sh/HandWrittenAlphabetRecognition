from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width,height,depth,classes):
        #initialize the model along with the input shape to be
        # "channel last"

        model=Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first ", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(32, (3,3), padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


