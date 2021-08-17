import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

def spike(x, threshold=1, bias=0, thresholding=0.5,scaling_factor=1,T=255,ext=0,noneloss=False):
    #x = tf.math.floordiv(x, v_thr, name=None)     
    _T = T + ext
    bias = tf.math.multiply(bias, _T, name=None)
    x = tf.math.add(x, bias, name=None)   
    x = tf.nn.relu(x)
    
    _t = scaling_factor*threshold    
    tre =  _t*thresholding * (not noneloss)
    x = tf.math.add(x,tre, name=None)
    x = tf.math.divide(x, _t, name=None)
    @tf.custom_gradient
    def custom_floor(x):
        def grad_fn(dy):
            return dy
        return tf.floor(x), grad_fn
    
    #x = tf.floor(x,name="Floor")
    pred0 = tf.constant(noneloss, dtype=tf.bool)
    x = tf.cond(pred0,lambda: x,lambda: custom_floor(x))
    
    pred1 = tf.constant(ext<0, dtype=tf.bool)
    x = tf.cond(pred1,lambda: x,lambda: tf.clip_by_value(x,0,_T))
    return x

get_custom_objects().update({'spike': Activation(spike)})

class SpikeActivation(Layer):
    def __init__(self, timesteps=255, threshold=1,bias=0, thresholding=0.5, scaling_factor=1,
                 spike_ext=0,noneloss=False,**kwargs):
        super(SpikeActivation, self).__init__(**kwargs)
        self.spike_ext = int(spike_ext)
        self.timesteps = int(timesteps)
        self.bias = K.cast_to_floatx(bias)        
        self.noneloss = noneloss
        self.threshold = K.cast_to_floatx(threshold)
        self.thresholding = K.cast_to_floatx(thresholding)
        self.scaling_factor = K.cast_to_floatx(scaling_factor)

    def call(self, inputs):
        return spike(inputs,
                     T=self.timesteps,
                     threshold=self.threshold,
                     bias=self.bias,
                     thresholding=self.thresholding,
                     ext=self.spike_ext,
                     noneloss=self.noneloss,
                     scaling_factor=self.scaling_factor) 

    def get_config(self):
        config = {'timesteps': int(self.timesteps),
                  'threshold': self.threshold,
                  'bias': self.bias,
                  'spike_ext': int(self.spike_ext),
                  'thresholding': self.thresholding,
                  'noneloss': self.noneloss,
                  'scaling_factor': self.scaling_factor}
        base_config = super(SpikeActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape