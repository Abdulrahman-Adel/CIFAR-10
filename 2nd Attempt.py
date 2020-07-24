from keras.datasets import cifar10
import matplotlib.pyplot as plt
import pandas as pd



(X_train,y_train),(X_test,y_test) = cifar10.load_data()

y_train = pd.Series(y_train.reshape(-1,))
y_test = pd.Series(y_test.reshape(-1,))

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

import random 

plt.imshow(random.choice(X_train))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train/255.0
X_test = X_test/255.0

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])

gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = gen.flow(X_train, y_train, batch_size=64)
steps = int(X_train.shape[0] / 64)

history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=100, validation_data=(X_test, y_test), verbose=0)


plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')

plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')

_, acc = model.evaluate(X_test,y_test, verbose = 0)
print('> %.3f' % (acc * 100.0))