from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Training data shape : ', train_images.shape, train_labels.shape)

print('Testing data shape : ', test_images.shape, test_labels.shape)

# for i in range(10):
#     cv2.imshow("dsfa", train_images[i])
#     cv2.waitKey()

# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

print(train_images)

dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

print(train_data)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

print(train_data)

train_data /= 255
test_data /= 255
print(train_data)
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

print('Original label 0: ', train_labels[0])
print('after categorical conv: ', train_labels_one_hot[0])

dropout_ratio = 0.5

model_reg = Sequential()
model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
model_reg.add(Dropout(dropout_ratio))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(dropout_ratio))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(dropout_ratio))
model_reg.add(Dense(nClasses, activation='softmax'))

model_reg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history_reg = model_reg.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                            validation_data=(test_data, test_labels_one_hot))
[reg_loss, reg_acc] = model_reg.evaluate(test_data, test_labels_one_hot)

print('Evaluation with regularization: Loss = {}, accuacy = {}'.format(reg_loss, reg_acc))


print(type(history_reg.history))
print(history_reg.history)


# Plot the Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history_reg.history['loss'], 'r', linewidth=3.0)
plt.plot(history_reg.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Reg Loss Curves', fontsize=16)

# Plot the Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history_reg.history['acc'], 'r', linewidth=3.0)
plt.plot(history_reg.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Reg Accuracy Curves', fontsize=16)

print(model_reg.predict_classes(test_data[[0], :]))
print(model_reg.predict(test_data[[0], :]))

plt.show()
