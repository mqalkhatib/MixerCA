'''
new code 30/10/2024
'''

from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import scipy
from utils import *
import pandas as pd
from keras import layers
from Attention_library import CA, SE, ECA, CBAM

## GLOBAL VARIABLES
dataset = 'GP' #input("select the data set:\nPU for pavia\nSA for Salinas\nIP for indian pines\nGP for GulfPort\n") 
Train_ratio = 0.01
windowSize = 13 
PCA_comp = 15

X, gt = loadData(dataset)


# Apply PCA for dimensionality reduction
X,pca = applyPCA(X,PCA_comp)


X, y = createImageCubes(X, gt, windowSize=windowSize)

Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, Train_ratio)


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    residuals = x
    
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    pos_emb4 = layers.Conv2D(filters, kernel_size=1)(x)
    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3, pos_emb4])

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])

    return x


def ConvMixerNet(img_list, NumClasses, depth, filters):
    inputs = keras.Input(img_list.shape[1:])
    
    x = layers.Conv2D(filters, kernel_size=1, strides=1)(inputs)
    x = activation_block(x)  

    # Apply the mixer blocks
    for ii in range(depth):
        x = mixer_block(x, filters, 3)

    
    x = CA(x)   
    # Global Pooling and Final Classification Block
    x = layers.GlobalAvgPool2D()(x)  # Global pooling in 2D
    logits = layers.Dense(NumClasses , activation="softmax")(x)

    model = keras.Model(inputs=[inputs], outputs=logits)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

Filters = 64
model= ConvMixerNet(Xtrain,num_classes(dataset), depth=4, filters=Filters)
model.summary()
from net_flops import net_flops
net_flops(model)


ytrain =keras.utils.to_categorical(ytrain)
ytest =keras.utils.to_categorical(ytest)


from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='loss', 
                              patience=10,
                              restore_best_weights=True
                              )


Aa = []
Oa = []
K = []
for i in range(10): 
    print("Iteration #", i)    
    model= ConvMixerNet(Xtrain,num_classes(dataset), depth=4, filters=Filters)
       
    history = model.fit(Xtrain, ytrain,
                        batch_size = 32, 
                        verbose=1, 
                        epochs=150, 
                        shuffle=True, 
                        callbacks = [early_stopper])
    
    #model.save_weights('./Models_Weights/'+ dataset +'/MixerCA_iter_' + str(i)+'.h5')
    
    #model.load_weights('./Models_Weights/'+ dataset +'/HybridSN/winSize_' + str(windowSize) + '_' + str(PCA_comp) + '_pca_model_' + str(i)+'.h5')

    
    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)

    
    
    
    kappa = cohen_kappa_score(np.argmax(ytest, axis=1),  y_pred_test)
    oa = accuracy_score(np.argmax(ytest, axis=1), y_pred_test)
    confusion = confusion_matrix(np.argmax(ytest, axis=1), y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
  
    Aa.append(float(format((aa)*100, ".2f")))
    Oa.append(float(format((oa)*100, ".2f")))
    K.append(float(format((kappa)*100, ".2f")))
    
 

print("oa = ", Oa) 
print("aa = ", Aa)
print('Kappa = ', K)
print('\n')
print('Mean OA = ', format(np.mean(Oa), ".2f"), '+', format(np.std(Oa), ".2f"))
print('Mean AA = ', format(np.mean(Aa), ".2f"), '+', format(np.std(Aa), ".2f"))
print('Mean Kappa = ', format(np.mean(K), ".2f"), '+', format(np.std(K), ".2f"))

'''
###############################################################################
# Create the predicted class map
del Xtrain, Xtest, ytrain, ytest
X, gt = loadData(dataset)
X,pca = applyPCA(X,PCA_comp)

X, y = createImageCubes(X, gt, windowSize, removeZeroLabels = False)

i = int(input("Enter Iteration number: "))
model.load_weights('./Models_Weights/'+ dataset +'/MixerCA_iter_' + str(i)+'.h5')
Y_pred_test = model.predict(X)
y_pred_test = (np.argmax(Y_pred_test, axis=1)).astype(np.uint8)

Y_pred = np.reshape(y_pred_test, gt.shape) + 1

sio.savemat('./Matlab Outputs/'+ dataset + '/Mixer_CA_full.mat', {'Mixer_CA_full': Y_pred})

gt_binary = gt

gt_binary[gt_binary>0]=1


new_map = Y_pred*gt_binary

sio.savemat('./Matlab Outputs/'+ dataset + '/Mixer_CA.mat', {'Mixer_CA': new_map})
'''


