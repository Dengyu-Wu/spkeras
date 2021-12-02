import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Conv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

def _Residual(inputs,filters,kernel_size):
    identity =  Activation("linear", trainable=False)(inputs)
    x = Conv2D( self.filters,
                self.kernel_size,
                padding="same")(identity)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D( self.filters,
                self.kernel_size,
                padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    residual = Add()([x, identity])
    x = Activation("relu")(residual)
    return x
    
class Residual(Layer):
    def __init__(self, filters,kernel_size,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def call(self, inputs):
        identity =  Activation("linear", trainable=False)(inputs)
        x = Conv2D( self.filters,
                    self.kernel_size,
                    padding="same")(identity)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D( self.filters,
                    self.kernel_size,
                    padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        residual = Add()([x, identity])
        x = Activation("relu")(residual)
        return x
    
    def get_config(self):
        config = {'filters': int(self.filters),
                  'kernel_size': self.kernel_size}
        base_config = super(Residual, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
