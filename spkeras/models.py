from tensorflow.keras import backend as K
from .layers import CurrentBias, SpikeForward

    
class cnn_to_snn(object):
    def __init__(self,timesteps=256,thresholding=0.5,signed_bit=0,scaling_factor=1,method=1,amp_factor=100
                 ,epsilon = 0.001, spike_ext=0,noneloss=False):
        self.timesteps = timesteps
        self.thresholding = thresholding
        self.scaling_factor = scaling_factor
        self.method = method
        self.epsilon = epsilon
        self.amp_factor = amp_factor
        self.bit = signed_bit
        self.spike_ext = spike_ext
        self.noneloss = noneloss
        self.use_bias = None
        
    def __call__(self,mdl,x_train):
        for layer in mdl.layers:
            layer_type = type(layer).__name__
            if hasattr(layer, 'activation') and layer_type != 'Activation':
                use_bias = layer.use_bias
                break
        self.use_bias = use_bias        
        self.get_config()
        self.model = self.convert(mdl,x_train,                    
                                  thresholding = self.thresholding,
                                  scaling_factor = self.scaling_factor,
                                  method = self.method,
                                  timesteps=self.timesteps)
        
        return self
    
    def convert(self,mdl,x_train,thresholding=0.5,scaling_factor=1,method=0,timesteps=256):
        #method: 0:threshold norm 1:weight norm 
        print('Start Converting...')
        from tensorflow.keras.models import Sequential, model_from_json
        from tensorflow.keras import activations
        import numpy as np
        
        model = model_from_json(mdl.to_json())
        model.set_weights(mdl.get_weights())
        
        snn_model = Sequential()
        epsilon = self.epsilon # defaut value in Keras
        amp_factor = self.amp_factor
        bit = self.bit - 1 if self.bit > 0 else False
        method = 1 if bit == True else method 
        spike_ext = self.spike_ext
        
        # Go through all layers, if it has a ReLU activation, replace it with PrELU
        l = self.findlambda(model, x_train, batch_size=100)
        
        k = 0 if model.layers[0].name == 'input' else 1
        
        def batchnormalization(weights,layer,amp_factor=100):
            gamma,beta,mean,variance = layer.get_weights()
            weights[0] = amp_factor*gamma/np.sqrt(variance+epsilon)*weights[0] 
            weights[1] = amp_factor*(gamma/np.sqrt(variance+epsilon)
                                             *(weights[1]-mean)+beta)
            return weights
                
        print('Building new model...')
        input_shape = model.layers[0].input_shape
        m = 0
        layer_num = 0
        for layer in model.layers:
            layer_type = type(layer).__name__            
            k = len(l)-1 if k > len(l)-1 else k   
            threshold = amp_factor*1 if method == 1 else amp_factor*l[k]/l[k-1]
                
            if hasattr(layer, 'activation') and layer_type != 'Activation':
                layer_type2 = type(model.layers[layer_num+1]).__name__
                
                snn_model.add(layer)                
                weights = layer.get_weights()                
                           
                if layer_type2 == 'BatchNormalization':
                    weights = batchnormalization(weights,model.layers[layer_num+1],amp_factor=amp_factor)
                else:
                    weights[0] = amp_factor*weights[0] 
                    if layer.use_bias:
                        weights[1] = amp_factor*weights[1]                           
            
                
                if bit:            
                    weights[0] = weights[0]/amp_factor
                    w_max = np.max(weights[0])
                    w_min = abs(np.min(weights[0]))
                    w_max = np.max([w_max,w_min])
                    print('maximum_weight:',w_max)
                    threshold = (1/(w_max))*2**bit*l[k]/l[k-1]
                    threshold = int(threshold)
                    weights[0] = (weights[0]*2**bit)/(w_max)
                    weights[0] = weights[0].astype(int)                    
                    if layer.use_bias:                    
                        weights[1] = (weights[1]*2**bit)/(amp_factor*w_max*l[k-1])
                        weights[1] = weights[1].astype(int)
                else:
                    weights[0] = weights[0]*l[k-1]/l[k] if method == 1 else weights[0]
                    if layer.use_bias: 
                        weights[1] = weights[1]/l[k]  if method == 1 else weights[1]/l[k-1]
                   
                
                snn_model.layers[-1].set_weights(weights)
                
                if layer.use_bias:
                    
                    bias = weights[1] 
                    weights[1] = 0*weights[1]
                    currentbias = CurrentBias(bias=bias,timesteps=timesteps,spike_ext=spike_ext)
                    snn_model.layers[-1].set_weights(weights)            
                    snn_model.add(currentbias)
                
                #add spike layer
                layer_name = "spikeforward_" + str(m)
                spikelayer = SpikeForward(threshold=threshold,
                                          thresholding=thresholding,
                                          scaling_factor=scaling_factor,
                                          timesteps=timesteps,
                                          spike_ext=spike_ext,
                                          name=layer_name)
                print('spikeforward_'+str(m)+'_threshold:',threshold)
                snn_model.add(spikelayer)
                m += 1
                k += 1
    
            elif layer_type == 'Flatten':
                snn_model.add(layer)
            elif hasattr(layer, 'pool_size'):
                threshold = l[k]/l[k-1]
                snn_model.add(layer)
                layer_name = "spikeforward_" + str(m)
                spikelayer = SpikeForward(threshold=threshold,
                                          thresholding=thresholding,
                                          timesteps=timesteps,
                                          scaling_factor=scaling_factor,
                                          spike_ext=spike_ext,
                                          name=layer_name)
                print('spikeforward_'+str(m)+'_threshold:',threshold)
                snn_model.add(spikelayer)
                m += 1
                k += 1
    
            elif layer_type == 'Activation' and layer.activation.__name__ == 'softmax':
                snn_model.add(layer)
                 
            layer_num += 1
            
        new_model = model_from_json(snn_model.to_json(),
                                     custom_objects={'SpikeForward':SpikeForward,
                                                     'CurrentBias':CurrentBias})
        new_model.build(input_shape)
        m = 0
        for layer in new_model.layers:
            layer.set_weights(snn_model.layers[m].get_weights())
            m += 1
        new_model.compile('adam', 'categorical_crossentropy', ['accuracy']) 
        del mdl
        print('New model generated!')
        return new_model        


    def findlambda(self,model,x_train,batch_size=100):   
        import numpy as np
        #k = 0
        lmax = np.max(x_train) 
        l = []
        if model.layers[0].name != 'input':
            l.append(lmax)
        print('Extracting Lambda...')#,end='')
        k = 0
        layer_num = len(model.layers)
        for layer in model.layers:
            layer_type = type(layer).__name__
            if hasattr(layer, 'activation') and layer_type == 'Activation' or hasattr(layer, 'pool_size'):
                print('{0}/{1}'.format(k,layer_num),end='')
                print(layer.__class__.__name__)
                if hasattr(layer, 'activation'):
                    if layer.activation.__name__ == 'softmax':
                        layer = model.layers[-2]
                    
                functor= K.function([model.layers[0].input], [layer.output])
                lmax = 0
                for n in range(x_train.shape[0]//batch_size):
                    a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                    #a =1 
                    _lmax = np.max(a)
                    lmax = max(lmax,_lmax)
                l.append(lmax)
            k += 1
        print('maximum activations:',l)
        return l    
    
    def SpikeCounter(self,x_train,timesteps=255,thresholding=1,scaling_factor=1,
                     spike_ext=0,batch_size=100,noneloss=False,mode=0):
        
        import numpy as np
        self.timesteps = timesteps
        self.thresholding = thresholding
        self.scaling_factor = scaling_factor
        self.spike_ext = spike_ext
        self.noneloss = noneloss
        self.model = self.chts_model(timesteps,thresholding,scaling_factor,spike_ext=spike_ext,noneloss=noneloss)
            
        self.get_config()    
        x_train = np.floor(x_train*timesteps)
        model = self.model
        
        cnt = []
        l = []
        print('Extracting Spikes...')#,end='')
        k = 0
        for layer in model.layers:
            #print('.',end='')
            layer_type = type(layer).__name__

            if layer_type == 'SpikeForward':
                print(layer.__class__.__name__)
                functor= K.function([model.layers[0].input], [layer.output])
                _cnt = []
                lmax = 0
                for n in range(x_train.shape[0]//batch_size):
                    a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                    if mode:                   
                        __cnt = np.floor(a)                      
                    else:
                        __cnt = np.sum(a)
                    _lmax = np.max(a)
                    lmax = max(lmax,_lmax)   
                    _cnt.append(__cnt)
                
                    
                if mode:
                    scnt = []
                    _cnt = np.array(_cnt)
                    n = int(np.max(_cnt))+1
                    for m in range(n):
                        scnt.append(np.count_nonzero((_cnt == m)))
                    _cnt = scnt
                else:
                    _cnt=np.ceil(np.sum(_cnt)/x_train.shape[0])
                    
                l.append(lmax)
                cnt.append(_cnt)
            k += 1
        print('Max Spikes for each layer:',l)  
        print('Total Spikes for each layer:',cnt)
        return l,cnt     
    
    def NeuronNumbers(self,mode=0): 
        #mode: 0. count every layer; 1. not count average pooling
        import numpy as np
        model = self.model
        k = 0
        cnt = []
        s = []
        print('Extracting NeuronNumbers...')#,end='')
        for layer in model.layers:
            #print('.',end='')
            layer_type = type(layer).__name__
            if layer_type == 'Conv2D' or layer_type == 'Dense':
                print(layer.__class__.__name__)
                s.append(layer.weights[0].shape)
                            
            if layer_type == 'SpikeForward': 
                print(layer.__class__.__name__)
                if hasattr(model.layers[k-1], 'pool_size') and mode == 1:
                    k +=1
                    continue
                    
                _cnt = np.prod(layer.output_shape[1:])
                cnt.append(_cnt)
            k += 1
        print('Total Neuron Number:',cnt)
        print('Done!')
        return s,cnt
           
    def evaluate(self,x_test,y_test,timesteps=256,thresholding=0.5,scaling_factor=1,spike_ext=0,noneloss=False):
        import numpy as np
        self.timesteps = timesteps
        self.thresholding = thresholding
        self.scaling_factor = scaling_factor
        self.spike_ext = spike_ext
        self.noneloss = noneloss
        self.model = self.chts_model(timesteps,thresholding,scaling_factor,spike_ext,noneloss)
        
        self.get_config()
        
        return self.model.evaluate(np.floor(x_test*self.timesteps),y_test)

    def chts_model(self,timesteps=256,thresholding=0.5,scaling_factor=1,spike_ext=0,noneloss=False):
        #method: 0:threshold norm 1:weight norm 
        from tensorflow.keras.models import Sequential, model_from_json
        from tensorflow.keras import activations
        mdl = self.model
        model = model_from_json(mdl.to_json(),
                               custom_objects={'SpikeForward':SpikeForward,
                                                'CurrentBias':CurrentBias})
        model.set_weights(mdl.get_weights())
        input_shape = model.layers[0].input_shape
        # Go through all layers, if it has a ReLU activation, replace it with PrELU
        print('Changing model timesteps...')        
        for layer in model.layers:
            layer_type = type(layer).__name__
            if layer_type == 'CurrentBias':
                layer.timesteps = timesteps
                layer.spike_ext = spike_ext
            if layer_type == 'SpikeForward':
                layer.thresholding = thresholding
                layer.scaling_factor = scaling_factor
                layer.timesteps = timesteps   
                layer.spike_ext = spike_ext
                layer.noneloss = noneloss
    
        new_model = model_from_json(model.to_json(),
                                     custom_objects={'SpikeForward':SpikeForward,
                                                     'CurrentBias':CurrentBias})
        new_model.build(input_shape)
        m = 0
        for layer in new_model.layers:
            layer.set_weights(mdl.layers[m].get_weights())
            m += 1
        new_model.compile('adam', 'categorical_crossentropy', ['accuracy']) 
        del mdl
        print('New model generated!')
        return new_model    
    
    def get_config(self):
        config = {'timesteps': int(self.timesteps),
                  'thresholding': self.thresholding,
                  'amp_factor':self.amp_factor, 
                  'signed_bit': self.bit,
                  'spike_ext':self.spike_ext,
                  'epsilon':self.epsilon,
                  'use_bias':self.use_bias,                               
                  'scaling_factor': self.scaling_factor,
                  'noneloss': self.noneloss,
                  'method':self.method
                  }
        return print(dict(list(config.items())))
       