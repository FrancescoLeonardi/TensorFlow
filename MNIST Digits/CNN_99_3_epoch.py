import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers




# Data preprocessing function
def preprocessing_data(x_train, x_test):

    def standardize(x, mean, std):
        return (x-mean)/std
    
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    x_train = standardize(x_train, np.mean(x_train), np.std(x_train))
    x_test = standardize(x_test, np.mean(x_test), np.std(x_test))
    
    return x_train, x_test


# Creation of CNN
def modelCNN(input_shape):

    input_ = keras.Input(input_shape)
    
    # BLOCK 1     
    cv01 = layers.Conv2D(64, kernel_size=(3, 3), activation="selu",
                         kernel_initializer=keras.initializers.HeUniform(),
                         kernel_regularizer=regularizers.l2(l2=0.0001))(input_)
    cv02 = layers.Conv2D(64, kernel_size=(3, 3), activation="selu",
                         kernel_initializer=keras.initializers.HeUniform(),
                         kernel_regularizer=regularizers.l2(l2=0.0001))(cv01)
    mp01 = layers.MaxPooling2D(pool_size=(2, 2))(cv02)                           
    bn01 = layers.BatchNormalization()(mp01)   
    
    # BLOCK 2
    cv03 = layers.Conv2D(128, kernel_size=(3, 3), activation="selu",
                         kernel_initializer=keras.initializers.HeUniform(),
                         kernel_regularizer=regularizers.l2(l2=0.0001))(bn01)
    cv04 = layers.Conv2D(128, kernel_size=(3, 3), activation="relu",
                         kernel_initializer=keras.initializers.HeUniform(),
                         kernel_regularizer=regularizers.l2(l2=0.0001))(cv03)
    mp02 = layers.MaxPooling2D(pool_size=(2, 2))(cv04)                           
    bn02 = layers.BatchNormalization()(mp02)                     
       
    # BLOCK 3 
    ft2 = layers.Flatten()(bn02)
    df1 = layers.Dense(256, activation="sigmoid")(ft2) 
    drop = layers.Dropout(0.75)(df1)    
    output = layers.Dense(10, activation="softmax")(drop)
    
    
    # MODEL  
    model = Model(input_, output) 
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model



# Print the accuracy and loss graph
def printGraphs(history):

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'], loc="upper left")
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','test'], loc="upper left")
    plt.show()
    
    



#---#---#         SEED         #---#---#

#tf.random.set_seed(2)
# seed 0 -> 99.22%
# seed 1 -> 99.33%
# seed 2 -> 99.07%
# random seed -> ?

#---#---#      PARAMETERS      #---#---#

epochs = 3
batch_size = 100
num_classes = 10
validation_split_rate = 0.2

input_shape = (28, 28, 1)



#---#---#         DATA         #---#---#

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = preprocessing_data(x_train, x_test)



#---#---#         MODEL         #---#---#    
    
model = modelCNN(input_shape)

history = model.fit(x_train, y_train,  
                    shuffle = True,  
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_split = validation_split_rate) 

score = model.evaluate(x_test, y_test, verbose=1)



printGraphs(history)

















