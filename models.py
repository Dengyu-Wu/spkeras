from tensorflow.keras import backend as K
from .layers import  SpikeActivation
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras import activations
    
class cnn_to_snn(object):
    def __init__(self,timesteps=256,thresholding=0.5,signed_bit=0,scaling_factor=1,method=1,amp_factor=100
                 ,epsilon = 0.001, spike_ext=0,noneloss=False,user_define=0):
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
        self.user_define = user_define
        
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
    
    def convert(self, mdl,x_train,thresholding=0.5,scaling_factor=1,method=0,timesteps=256):
        print('Start Converting...')
        from tensorflow.keras.models import Sequential, model_from_json
        from tensorflow.keras import activations
        import numpy as np
        import json
        
        old_mdl = mdl.to_json()
        new_mdl = mdl.to_json()

        old_mdl = json.loads(old_mdl)
        new_mdl = json.loads(new_mdl)

        model = model_from_json(mdl.to_json())
        
        model.set_weights(mdl.get_weights())

        epsilon = self.epsilon # defaut value in Keras
        amp_factor = self.amp_factor
        bit = self.bit - 1 if self.bit > 0 else 0
        method = 1 if bit == True else method 
        spike_ext = self.spike_ext
        user_define = self.user_define
        
        l,lmax = self.findlambda(model, x_train, batch_size=100,user_define=user_define)       
        layers=[]                            
        weights = []
        for layer in model.layers:
            _weights = layer.get_weights()
            layer_type = type(layer).__name__
            if _weights != [] and hasattr(layer, 'activation') :
                if not layer.use_bias:
                    _weights = [_weights,0] 
                _weights = [_weights,1] 
                weights.append(_weights)
            if layer_type == 'AveragePooling2D':
                weights.append([0,0])
            if layer_type == 'BatchNormalization':
                gamma,beta,mean,variance = layer.get_weights()
                weights[-1][0][0] = gamma/np.sqrt(variance+epsilon)*weights[-1][0][0] 
                weights[-1][0][1] = (gamma/np.sqrt(variance+epsilon)
                                                 *(weights[-1][0][1]-mean)+beta)
            
                
        vthr = []
        bias=[]
        kappa = amp_factor
        new_weights = []
        num=0
        for _weights in weights:
            if _weights[1] == 0:
                vthr.append(l[num].tolist()) 
                bias.append(0) 
                num += 1
                continue
                
            _weights = _weights[0]
            norm = 1
            if bit > 0:
                norm = np.max(np.abs(_weights[0]))
                _weights[0] = _weights[0]/norm*2**bit
                _weights[0] = _weights[0].astype(int)   
                _weights[0] = _weights[0]/2**bit
            _bias = kappa*_weights[1]/lmax[num+1]
            _bias = _bias/norm
            bias.append(_bias.tolist())    
            _weights[0] = kappa*_weights[0]/l[num]
            _weights[1] = _weights[1]*0
            new_weights.append(_weights)
            vthr.append(norm*kappa) 
            num += 1
        weights = new_weights
            
        print('Number of Spiking Layer:',len(vthr))
        print('threshold:',vthr)
        print('Note: threshold will be different when weight quantisation applied!')
        #n=2                           
        num = 0
        loc = 0
        functional = old_mdl['class_name'] == 'Functional'
        inbound_nodes = False
        for layer in old_mdl['config']['layers']:
            if layer['class_name'] == 'InputLayer':
                layers.append(layer)
            if 'activation' in layer['config'] and layer['class_name'] != 'Activation':
                if functional:
                    inbound_nodes=layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]]
                    inbound_nodes = layer['config']['name']
                layers.append(layer)
                layers.append(self.spike_activation(threshold=vthr[loc],bias=bias[loc],scaling_factor=kappa,inbound_nodes=inbound_nodes,name='spike_activation_'+str(loc)))   
                loc += 1
            if layer['class_name']=='Add':
                l_num = []
                for _inputs in layer['inbound_nodes'][0]:
                    txt = _inputs[0]
                    _txt = txt.split('_')
                    txt = 0 if _txt[0] == _txt[-1] else _txt[-1]
                    l_num.append(txt)

                l_gap = abs(int(l_num[0])-int(l_num[1]))*2
                identity=layers[-1-l_gap]['config']['name']
                residual=layers[-1]['config']['name']
                inbound_node=[[[identity, 0, 0, {}], [residual, 0, 0, {}]]]
                layer['inbound_nodes']=inbound_node
                layers.append(layer)
            if layer['class_name']=='Flatten':
                if functional:
                    inbound_nodes=layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]]
                layers.append(layer)
            if layer['class_name']=='AveragePooling2D':
                if functional:
                    inbound_nodes = layers[-1]['config']['name']
                    layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]] 
                    inbound_nodes = layer['config']['name']
                layers.append(layer)
                layers.append(self.spike_activation(threshold=vthr[loc],bias=bias[loc],scaling_factor=kappa,inbound_nodes=inbound_nodes,name='spike_activation_'+str(loc)))
                loc += 1
            if layer['class_name']=='Activation':
                if layer['config']['activation'] == 'softmax':
                    if functional:
                        inbound_nodes = layers[-1]['config']['name']
                        layer['inbound_nodes'] = [[[inbound_nodes, 0, 0, {}]]] 
                        inbound_nodes = layer['config']['name']    
                    layers.append(layer)
            num+=1
        new_mdl['config']['layers'] = layers
        new_mdl = json.dumps(new_mdl)
        new_model = model_from_json(new_mdl,
                                     custom_objects={'SpikeActivation':SpikeActivation})
        input_shape = model.layers[0].input_shape
        #new_model.build(input_shape)                            
        #new_model = keras.Model(inputs=inputs, outputs=outputs)

        m = 0
        for layer in new_model.layers:
            layer_type = type(layer).__name__ 
            if hasattr(layer, 'activation') and layer_type != 'Activation':
                if layer.use_bias:
                    layer.set_weights(weights[m])
                else:
                    layer.set_weights(weights[m][0])
                m += 1
        new_model.compile('adam', 'categorical_crossentropy', ['accuracy']) 
        print('New model generated!')
        return new_model

    def spike_activation(self,threshold, bias,timesteps=255, spike_ext=0,scaling_factor=1,inbound_nodes=False,name='spike_activation'):
        layer_config = {'class_name': 'SpikeActivation',
         'config': {'name': 'spike_activation_1',
          'trainable': True,
          'dtype': 'float32',
          'timesteps': 255,
          'threshold': 1.0,
          'spike_ext': 0,
          'thresholding': 0.5,
          'noneloss': False,
          'scaling_factor': 1.0},
          'name': 'spike_activation_1'}    

        if not inbound_nodes:
            layer_config['name']=name
            layer_config['config']['name']=name
            layer_config['config']['timesteps']=timesteps
            layer_config['config']['threshold']=threshold
            layer_config['config']['bias']=bias
            layer_config['config']['spike_ext']=spike_ext
            layer_config['scaling_factor']=scaling_factor
        else:
            layer_config['name']=name
            layer_config['config']['name']=name
            layer_config['config']['timesteps']=timesteps
            layer_config['config']['threshold']=threshold
            layer_config['config']['bias']=bias
            layer_config['config']['spike_ext']=spike_ext
            layer_config['config']['scaling_factor']=scaling_factor 
            layer_config['inbound_nodes']=[[[inbound_nodes, 0, 0, {}]]] 
        return layer_config
    
    def findlambda(self,model,x_train,batch_size=100,user_define=False):  
        import numpy as np
        #k = 0
        lmax = np.max(x_train) 
        l = []
        if model.layers[0].name != 'input':
            l.append(lmax)
        print('Extracting Lambda...')#,end='')
        k = 0
        layer_num = len(model.layers)
        act_num_max = 0
        add_cnt = 2
        for layer in model.layers:
            layer_type = type(layer).__name__
            if hasattr(layer, 'activation') and layer_type == 'Activation' or hasattr(layer, 'pool_size'):
                IsAddLayer = type(model.layers[k-1]).__name__ == 'Add' 
                if IsAddLayer:
                    layer = model.layers[k-1]
                    print('{0}/{1}'.format(k,layer_num),end='')
                    print(layer.__class__.__name__) 
                    l_num = []
                    for _inputs in layer.input:
                        txt = _inputs.name
                        txt = txt.split('/')[0]
                        _txt = txt.split('_')
                        txt = 0 if _txt[0] == _txt[-1] else _txt[-1]
                        l_num.append(txt)
                    act_num_min = min(int(l_num[0]),int(l_num[1]))
                    l_gap = abs(int(l_num[0])-int(l_num[1]))
                    if not user_define:                       
                        functor= K.function([model.layers[0].input], [layer.output])
                        lmax = 0
                        for n in range(x_train.shape[0]//batch_size):
                            a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                            #a =1 
                            _lmax = np.max(a)
                            lmax = max(lmax,_lmax)   
                    else:
                        lmax = user_define   
                        
                    AddIsNeibour = (act_num_min - act_num_max) == 1
                    if AddIsNeibour:
                        add_cnt += 1
                    else:
                        add_cnt = 2
                        
                    for cnt in range(add_cnt):
                        l[-1-cnt*l_gap] = lmax                        
                    k+=1
                    
                    act_num_max = max(int(l_num[0]),int(l_num[1]))
                    continue
                    
                print('{0}/{1}'.format(k,layer_num),end='')
                print(layer.__class__.__name__)
                if hasattr(layer, 'activation'):
                    try:
                        _name = layer.activation.__name__
                    except:
                        _name = None
                    if _name == 'softmax':
                        layer = model.layers[-2]
                        functor= K.function([model.layers[0].input], [layer.output])
                        lmax = 0
                        for n in range(x_train.shape[0]//batch_size):
                            a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                            #a =1 
                            _lmax = np.max(a)
                            lmax = max(lmax,_lmax)
                        l.append(lmax)
                        continue

                if not user_define:                       
                    functor= K.function([model.layers[0].input], [layer.output])
                    lmax = 0
                    for n in range(x_train.shape[0]//batch_size):
                        a = functor(x_train[n*batch_size:(n+1)*batch_size])[0]
                        #a =1 
                        _lmax = np.max(a)
                        lmax = max(lmax,_lmax)
                else:
                    lmax = user_define
                l.append(lmax)
                
            k += 1
        print('maximum activations:',l)
        new_l = []
        num = 0
        for k in range(len(l)):
            if k == 0:
                continue
            new_l.append(l[k]/l[k-1])
        print('normalisation factor:',new_l)        
        return [new_l,l]
    
    def SpikeCounter(self,x_train,timesteps=255,thresholding=0.5,scaling_factor=1,
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
           
    def evaluate(self,x_test,y_test,timesteps=256,thresholding=0.5,scaling_factor=1,spike_ext=0,noneloss=False,sf=None,fix=0):
        import numpy as np
        self.timesteps = timesteps
        self.thresholding = thresholding
        self.scaling_factor = scaling_factor
        self.spike_ext = spike_ext
        self.noneloss = noneloss
        self.model = self.chts_model(timesteps=timesteps,thresholding=thresholding,
                                     scaling_factor=scaling_factor,
                                     spike_ext=spike_ext,noneloss=noneloss,sf=sf)
        
        self.get_config()
        _x_test = x_test*fix if fix > 0 else np.floor(x_test*self.timesteps)
        return self.model.evaluate(_x_test,y_test)

    def chts_model(self,timesteps=256,thresholding=0.5,scaling_factor=1,spike_ext=0,noneloss=False,sf=None):
        #method: 0:threshold norm 1:weight norm 
        from tensorflow.keras.models import Sequential, model_from_json
        from tensorflow.keras import activations
        mdl = self.model
        model = model_from_json(mdl.to_json(),
                                     custom_objects={'SpikeActivation':SpikeActivation})
        
        model.set_weights(mdl.get_weights())
        input_shape = model.layers[0].input_shape
        # Go through all layers, if it has a ReLU activation, replace it with PrELU
        print('Changing model timesteps...')
        k = 0
        
        for layer in model.layers:
            layer_type = type(layer).__name__
            if layer_type == 'SpikeActivation':
                layer.thresholding = thresholding
                layer.scaling_factor = scaling_factor # if sf == None else sf[k]
                layer.timesteps = timesteps   
                layer.spike_ext = spike_ext
                layer.noneloss = noneloss
                k += 1 
            
        new_model = model_from_json(model.to_json(),
                                     custom_objects={'SpikeActivation':SpikeActivation})
        #new_model.build(input_shape)
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
       
