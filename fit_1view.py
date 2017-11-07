
# coding: utf-8

import numpy
import ModelNet40Loader
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout

#nb_train_samples = 8000#9443
#nb_validation_samples = 400
epochs = 16
batch_size = 16

# build the ResNet50 network
base_model = applications.ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3))

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(40, activation='softmax'))
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# set 0.7 fraction of layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:(int)(len(model.layers)*0.7)]:
    layer.trainable = False


#print (len(model.layers))

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

(train_generator, train_dataset_size)=ModelNet40Loader.modelnet40_generator(
    subset='train',
    src_dir='ModelNet-40'
    )

(validation_generator, validation_dataset_size)=ModelNet40Loader.modelnet40_generator(
    subset='test',
    src_dir='ModelNet-40'
    )

#ImageDataGenerator
datagen = ImageDataGenerator(horizontal_flip=True)

#batching generator and image augmentation
def batching_generator(generator,batchsize):
    while(True):
        input_data=numpy.empty((batchsize,224,224,3))
        output_data=numpy.empty((batchsize,40))
        i=0;
        while(i<batchsize):
            (x,y)=generator.next()
            (x,y)=datagen.flow(x,y).next()
            input_data[i]=x
            output_data[i]=y
            i+=1
        yield input_data,output_data

model.fit_generator( 
    batching_generator(train_generator,batch_size),
    #train_generator,
    samples_per_epoch=train_dataset_size,
    epochs=epochs,
    #validation_data=validation_generator,
    validation_data=batching_generator(validation_generator,batch_size),
    nb_val_samples=validation_dataset_size)
