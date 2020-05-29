# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential() #ann initialize etmek için

# Step 1 - Convolution
#ilk konvolüsyon katmanı
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#pooling katmanı ile 2x2lik matrisleri tarayarak en büyük değeri sonraki katmana aktarıyoruz ama doğruluk düşüyor işlem hızı artıyor
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#2 yolla accuracy artar 1-ekstra convolusyon 2-ekstra fully connected böylece kolaylıkla %80 accuracy yakalanabilir
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
#çok boyutu klasik ann yapısına uygulanabilmesi için tek boyuta düşürüyoruz
classifier.add(Flatten())

# Step 4 - Full connection
#öğrenme ilemi fully connectedta gerçekleşir bundan öncesi feature learning
classifier.add(Dense(units = 128, activation = 'relu'))
#burada softmax ile yüzde kaç ne onu söyler
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), #input_shape ile aynı olacak
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64), #input_shape ile aynı olacak
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,    #training sayımız
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000) #test sayımız

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) #bir dimension daha ekliyoruz
result = classifier.predict(test_image)
training_set.class_indices #hangi sonuç neye işaret ediyor onu söyler

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
""" %90 accuracy solution
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
import os
 
 
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
 
script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(script_dir, '../dataset/training_set')
test_set_path = os.path.join(script_dir, '../dataset/test_set')
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')
 
test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='binary')
 
# Create a loss history
history = LossHistory()
 
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=90,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=12,
                         max_q_size=100,
                         callbacks=[history])
 
 
# Save model
model_backup_path = os.path.join(script_dir, '../dataset/cat_or_dogs_model.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)
 
# Save loss history to file
loss_history_path = os.path.join(script_dir, '../loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are:", training_set.class_indices)
loss: 0.1318 - acc: 0.9473 - val_loss: 0.3368 - val_acc: 0.9042"""