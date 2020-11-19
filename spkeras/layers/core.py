import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

def current_bias(x,b,T=255,ext=0):
    #x = tf.math.floordiv(x, v_thr, name=None)
    _T = T + ext
    b = tf.math.multiply(b, _T, name=None)
    x = tf.math.add(x, b, name=None)
    
    return x

get_custom_objects().update({'current_bias': Activation(current_bias)})

class CurrentBias(Layer):
    def __init__(self, bias=0,timesteps=255,spike_ext=0, **kwargs):
        super(CurrentBias, self).__init__(**kwargs)
        self.spike_ext = int(spike_ext)
        self.bias = K.cast_to_floatx(bias)
        self.timesteps = int(timesteps)

    def call(self, inputs):
        return current_bias(inputs,b=self.bias, T=self.timesteps,ext=self.spike_ext) 

    def get_config(self):
        config = {'bias': self.bias,
                  'spike_ext': int(self.spike_ext),
                  'timesteps': int(self.timesteps)}
        base_config = super(CurrentBias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def spike(x, threshold=1, thresholding=0.5,scaling_factor=1,T=255,ext=0,noneloss=False):
    #x = tf.math.floordiv(x, v_thr, name=None)
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
    
    _T = T + ext
    pred1 = tf.constant(ext<0, dtype=tf.bool)
    x = tf.cond(pred1,lambda: x,lambda: tf.clip_by_value(x,0,_T))
    return x

get_custom_objects().update({'spike': Activation(spike)})

class SpikeForward(Layer):
    def __init__(self, timesteps=255, threshold=1, thresholding=0.5, scaling_factor=1,
                 spike_ext=0,noneloss=False,**kwargs):
        super(SpikeForward, self).__init__(**kwargs)
        self.spike_ext = int(spike_ext)
        self.timesteps = int(timesteps)
        self.noneloss = noneloss
        self.threshold = K.cast_to_floatx(threshold)
        self.thresholding = K.cast_to_floatx(thresholding)
        self.scaling_factor = K.cast_to_floatx(scaling_factor)

    def call(self, inputs):
        return spike(inputs,
                     T=self.timesteps,
                     threshold=self.threshold,
                     thresholding=self.thresholding,
                     ext=self.spike_ext,
                     noneloss=self.noneloss,
                     scaling_factor=self.scaling_factor) 

    def get_config(self):
        config = {'timesteps': int(self.timesteps),
                  'threshold': self.threshold,
                  'spike_ext': int(self.spike_ext),
                  'thresholding': self.thresholding,
                  'noneloss': self.noneloss,
                  'scaling_factor': self.scaling_factor}
        base_config = super(SpikeForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
