from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras.layers import MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg19 import decode_predictions
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.utils.vis_utils import plot_model
import h5py
from tensorflow.python.keras.models import model_from_json
import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.python.keras.models import load_model
import cv2
import numpy

#import torchvision.datasets.imagenet as imagenet  #to include models like Squeezenet,Alexnet

model = vgg19.VGG19( weights = "imagenet", include_top = False, input_shape = (224,224,3))
#model.cuda()
# add new classifier layers
x = model.output
x = MaxPooling2D()(x)
for layer in model.layers:
    layer.trainable = False
x = Dense(units=256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.4)(x)
x = Dense(units=256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.4)(x)
x = Flatten()(x)
output = Dense(units=4, activation="softmax")(x)


model = Model(inputs=model.input, outputs=output)

# summarize
model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#optimizer=adam

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,#Generate batches of tensor image data with real-time data augmentation.
                                    rescale=1. / 255,#we multiply the data by the value provided (after applying all other transformations)
                                    zoom_range=0.2)# Float...Range for random zoom...If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]




train_generator = data_generator.flow_from_directory(
    "D:/PYCodes/Arrow224/train",
    target_size=(image_size, image_size),
    batch_size=32,#16
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True, )

validation_generator = data_generator.flow_from_directory(
    "D:/PYCodes/Arrow224/test",
    target_size=(image_size, image_size),
    class_mode='categorical',
    color_mode='rgb',
)

history=model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=3)
    
#To see trained model plots   
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.show()
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

#To save Model weights
model.save("weights/a14.h5")
print("Loaded model from disk")
print("Saved model to disk")
