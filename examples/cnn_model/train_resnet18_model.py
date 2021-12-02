from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Input
from tensorflow.keras import activations

from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, add
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

def Residual(inputs,filters,kernel_size,dropout):
    x = Conv2D( filters,
                kernel_size,
                padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Activation("relu")(x)
    x = Conv2D( filters,
                kernel_size,
                padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Activation("relu")(x)
    x = add([x, inputs])
    return Activation("relu")(x)

class build_model:
    def __init__(self,train=True):
        self.num_classes = 10
        self.weight_decay = 0.0001
        self.x_shape = [32,32,3]
        
        self.initializer = initializers.RandomNormal(mean=0.0, stddev=0.02895, seed=None)
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10_vgg16.h5')
            self.model = self.train(self.model)

    def extract_model(self):
        return self.model
    
    def build_model(self):
         # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        use_bias = True
        weight_decay = self.weight_decay
        inputs = Input(self.x_shape)
        
        x = Conv2D(64, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Activation("relu")(x)
        x = Residual(x,64,(3,3),0.3)
        x = Conv2D(128, (3, 3),strides=(2,2), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Activation("relu")(x)
        x = Conv2D(128, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Activation("relu")(x)        
        x = Residual(x,128,(3,3),0.3)
        x = Conv2D(256, (3, 3),strides=(2,2),  padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Activation("relu")(x)
        x = Conv2D(256, (3, 3),  padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Activation("relu")(x)        
        x = Residual(x,256,(3,3),0.4) 
        x = Conv2D(512, (3, 3),strides=(2,2),  padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Activation("relu")(x)
        x = Conv2D(512, (3, 3),  padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Activation("relu")(x)        
        x = Residual(x,512,(3,3),0.4)    
        x = AveragePooling2D(pool_size=(4,4),
                                 strides=(1, 1))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(self.num_classes,use_bias=use_bias,
                        kernel_regularizer=regularizers.l2(weight_decay))(x)
        outputs = Activation('softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="test")
        return model

    def train(self,model):
        #training parameters
        batch_size = 128
        maxepoches = 300
        learning_rate = 0.1
        lr_decay = 1e-6
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train/255
        x_test = x_test/255       
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        lr_decayed_fn = optimizers.schedules.CosineDecay(learning_rate, maxepoches) 
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_decayed_fn)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        
        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            
        # training process in a for loop with learning rate drop every 25 epoches.
        historytemp = model.fit(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=maxepoches,
                        validation_data=(x_test, y_test),verbose=2,callbacks=[reduce_lr])   
        save_model(model,'./cifar10_resnet.h5')
        return model

if __name__ == '__main__':
    model = build_model(train=1)
    mdl = model.extract_model()
