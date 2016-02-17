# Keras code for confocal laser scanner microscopy skin lesion classification.
# M.D. Bloice, 
# Inst. for Medical Informatics,
# Medical University Graz.
# <marcus.bloice@medunigraz.at>
# Portions of code from the Keras GitHub repository.
# https://github.com/fchollet/keras

'''
    Train a Convolutional Neural Network on a binary class CLSM lesion image dataset.
'''

import numpy as np
import glob
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
from terminaltables import AsciiTable
import random
import seaborn as sns

# Set this so that we can redo experiments.
np.random.seed(376483)

batch_size = 128              # Size of image batch during training
nb_classes = 2                # How many classes of objects do we have
nb_epoch = 20                 # Adjust until stabilising
img_rows, img_cols = 64, 64   # Input image dimensions
nb_filters = 32               # Number of convolutional filters to use
nb_pool = 2                   # Size of pooling area for max pooling
nb_conv = 3                   # Convolution kernel size

# This function gets all BMP images in the directory passed as an argument
# and returns the images as a matrix X and their labels, specified as an int,
# as a vector, y.
def loadImageData(path, label):
   listOfFiles = glob.glob(path)
   X = np.ndarray((len(listOfFiles), img_rows, img_cols))
   y = []
   imageNumber = 0
   for file in listOfFiles:
       current = io.imread(file)
       current = current[:,:,0] # for greyscale images, use [:,:,0]
       X[imageNumber,:,:] = current
       y.append(label)
       imageNumber += 1
   return X, y

# Define the paths as globs for the training sets
path_mm_test = '/path/to/malignantmelanoma/test/*.BMP'
path_mm_train = '/path/to/malignantmelanoma/train/*.BMP'
path_nz_train = '/path/to/benignnevi/train/*.BMP' 
path_nz_test = '/path/to/benignnevi/test/*.BMP'  

# Fetch the data and format it into Theano compatible matrices using the helper function above.
X_nz_train, y_nz_train = loadImageData(path_nz_train, 0)
X_nz_test, y_nz_test = loadImageData(path_nz_test, 0)
X_mm_train, y_mm_train = loadImageData(path_mm_train, 1)
X_mm_test, y_mm_test = loadImageData(path_mm_test, 1)

# Merge the seperate training sets into one, likewise for the labels
X_train = np.concatenate((X_mm_train, X_nz_train))
y_train = y_mm_train + y_nz_train

X_test = np.concatenate((X_mm_test, X_nz_test))
y_test = y_mm_test + y_nz_test

X_train = X[:5000]
y_train = y[:5000]
X_test = X[5000:]
y_test = y[5000:]

# Now reshape the data in the tensor format required by Keras/Theano.
# Also, use floating point values instaed of unsigned 8-bit integers 
# for better accuracy.
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

print str(X_train.shape[0]) + ' training samples.'
print str(X_test.shape[0]) + ' test samples.'

# This turns 0 into [1,0,0,0] 1 into [0,1,0,0] and 2 into [0,0,1,0] and so on...
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Begin building the model. A convolutional network is sequential. 
model = Sequential() 

# Add 2 convolutional/max-pooling pairs, use the Rectifed Linear Unit activation function and 50% dropout.
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='full', input_shape=(1, img_rows, img_cols)))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# After the convolutional and max-pooling layers, we flatten and create
# two fully connected layers, again using ReLU activation functions and 
# finally use the softmax function for prediction output. 
# To control overfitting we agressively use dropout once again.
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Compile our model into Theano code.
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

# Start training. 
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, 
                    verbose=1, validation_data=(X_test, Y_test))

# Once finished, we can score our model and print the overall accuracy.
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# For more details regarding the model's accuracy, we can compute
# a number of other metrics. For this we need the predicted labels
# and true labels for each image in the test set:
y_pred = model.predict_classes(X_test) # This will take a few seconds...
y_true = np_utils.categorical_probas_to_classes(Y_test)

target_names = ["Benign", "Malignant"]

# Compute a confusion matrix and a normalised confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# These can also be used to produce heatmaps using Matplotlib or Seaborn.
print(cm)
print(cm_normalised)

# Print a classification report, including precision, recall, and f1-score.
print(classification_report(y_true, y_pred, target_names=target_names)) 

np.savetxt('y_true.txt', y_true)
np.savetxt('y_pred.txt', y_pred)

# Plot a confusion matrix graphically 
sns.set(font_scale=4.5) 
fig, ax = plt.subplots(figsize=(30,20))
ax = sns.heatmap(cm, annot=True, linewidths=2.5, square=True, linecolor="Green", cmap="Greens", yticklabels=target_names, xticklabels=target_names, vmin=0, vmax=900, fmt="d", annot_kws={"size": 50})
ax.set(xlabel='Predicted label', ylabel='True label')

# Get values for plotting:
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

# Plots
fig, ax1 = plt.subplots()
plt.grid(True)
ax1.plot(acc, 'g-', linewidth=2.0, label="Accuracy")
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='g')
for tl in ax1.get_yticklabels():
    tl.set_color('g')
ax2 = ax1.twinx()
# Here we plot a point at 0,0, give it the label Accuracy and add the legend... it's a hack to get 2 labels
ax2.plot(0, 0, 'g-', label="Accuracy", linewidth=2.0)
ax2.plot(loss, 'r-', linewidth=2.0, label="Loss")
ax2.set_ylabel('Loss', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.xlim((1,49))
ax2.legend(loc='center right')
plt.show()

# END
