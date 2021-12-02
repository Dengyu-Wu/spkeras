from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import initializers



class build_model:
    def __init__(self,train=True):
        self.num_classes = 10
        self.weight_decay = 0.0001
        self.x_shape = [32,32,3]
        
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cnn_mdl..h5')
            self.model = self.train(self.model)

    def extract_model(self):
        return self.model
            
    def build_model(self):
        use_bias = True
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',use_bias=use_bias,
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(AveragePooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        model.add(Conv2D(512, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))


        model.add(AveragePooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        model.add(Conv2D(512, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same',use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,use_bias=use_bias,
                        kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes,use_bias=use_bias,
                        kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('softmax'))
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

        lr_decayed_fn = optimizers.schedules.CosineDecay(learning_rate,maxepoches) 
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
        save_model(model,'./cnn_mdl.h5')
        return model

if __name__ == '__main__':
    model = build_model(train=1)
