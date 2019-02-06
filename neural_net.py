from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np

def process(img):
    
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img2, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    contourAreaArray = [cv2.contourArea(c) for c in contours]

    if len(contourAreaArray) == 0:
        return img

    contourIndex = np.argmax(contourAreaArray)

    [x, y, w, h] = cv2.boundingRect(contours[contourIndex])

    procedImg = img[y:y+h+1, x:x+w+1]
	
    procedImg = cv2.resize(procedImg, (28, 28), interpolation=cv2.INTER_AREA)

    return procedImg


def prepareImages(images):

    for i in range(len(images)):
        print('\rPreparing ' + str(i) + '/' + str(len(images) - 1) , end='     ')
        images[i] = process(images[i])
    return images


def trainNN():
    
    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

    print('Preparing training data \n')
    trainImages = prepareImages(trainImages)

    print('Preparing test data \n')
    testImages = prepareImages(testImages)
	
    dimData = np.prod(trainImages.shape[1:])
    trainData = trainImages.reshape(trainImages.shape[0], dimData)
    testData = testImages.reshape(testImages.shape[0], dimData)

    nClasses = len(np.unique(trainLabels))

    trainData = np.array(trainData, 'float32') / 255
    testData = np.array(testData, 'float32') / 255

    trainLabelsOneHot = to_categorical(trainLabels)
    testLabelsOneHot = to_categorical(testLabels)


    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape = (dimData,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(trainData, trainLabelsOneHot, batch_size = 256, epochs = 20, verbose = 1, validation_data = (testData, testLabelsOneHot))
    
    [testLoss, testAcc] = model.evaluate(testData, testLabelsOneHot)

    print('Evaulation result on Test Data : Loss = {}, accuracy = {}'.format(testLoss, testAcc))

    print('Training data shape : ', trainImages.shape, trainLabels.shape)
 
    print('Testing data shape : ', testImages.shape, testImages.shape)
    
    # Find the unique numbers from the train labels
    classes = np.unique(trainLabels)
    nClasses = len(classes)
    print('Total number of outputs : ', nClasses)
    print('Output classes : ', classes)
    
    model.save('C:/soft/SoftComputing/model.h5')

    del model

#trainNN()
