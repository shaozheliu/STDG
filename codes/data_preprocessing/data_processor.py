from eegmodel.EEGModels import *
from DataLoader.DataLoader import *
import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import sys
K.set_image_data_format('channels_first')
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


TRAIN_MODEL = True
INFER_TEST = False
NUM_OF_FOLDS = 2
nb_classes = 4
channels = 22
samples = 751
batch_size = 64
EPOCH = 300
VERBOSE = 2
DATA_PATH = "../../Data/amex-data/BCICIV_2a_gdf/"
INFER_LABEL_PATH = "../../Data/amex-data/true_labels_2a/"
MODEL_SAVE_PATH = "./log/Cross-subjects/"

if TRAIN_MODEL:
    # outfile = open('log/exp_EEG_0.1.txt', 'w')
    # sys.stdout = outfile
    acc = []
    ka = []
    prec = []
    recall = []
    roc = []
    dataidx = [i for i in range(0,10)]
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(dataidx):
        print(f'train index: {train_idx}, test index: {test_idx}')
        X_test, y_test = load_BCI2a_data(file_name=f'{DATA_PATH}A0{test_idx}T.gdf')
        for k in range(len(train_idx)):
            if k == 0:
                X, y = load_BCI2a_data(file_name=f'{DATA_PATH}A0{train_idx[k]}T.gdf')
            else:
                tempdata, templabel = load_BCI2a_data(file_name=f'{DATA_PATH}A0{train_idx[k]}T.gdf')
                X = np.concatenate((X,tempdata),axis=0)
                y = np.concatenate((y,templabel),axis = 0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    # OneHotEncoding La bels
        enc = OneHotEncoder()
        y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()
        y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
        y_val = enc.fit_transform(y_val.reshape(-1, 1)).toarray()
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        X_val = np.expand_dims(X_val,axis=1)
        nn = EEGNet(nb_classes = nb_classes,Chans = channels,Samples = samples)
        opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
        my_callbacks = [
            # RocAucMetricCallback(),  # include it before EarlyStopping!
            EarlyStopping(monitor='val_accuracy', patience=20, verbose=2, mode='max')
        ]
        # 仅保存参数
        checkpointer = ModelCheckpoint(MODEL_SAVE_PATH + f"EEGNET_{test_idx}"+"_{epoch:02d}-{val_accuracy:.2f}.h5",
                                       monitor='val_accuracy',
                                       verbose=VERBOSE,
                                       save_best_only = True,
                                       save_weights_only = True)

        nn.compile(loss='categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
        nn.summary()
        nn.fit(X_train, y_train,validation_data=(X_val,y_val) ,batch_size=batch_size, epochs=EPOCH,callbacks=[checkpointer])
        y_pred = nn.predict(X_test)
        pred = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)

        acc.append(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
        ka.append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
        prec.append(precision_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))
        recall.append(recall_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))
        roc.append(recall_score(y_test.argmax(axis = 1),pred.argmax(axis=1), average='weighted'))
    ## 输出所有结果log
    outputfile = open(f'{MODEL_SAVE_PATH}EEGNet_output.txt', mode='w')
    outputfile.write(f'The accuracy is: {acc}, cross-subject acc is: {np.mean(acc)} \n')
    outputfile.write(f'The acohen_kappa_score is: {ka}, cross-subject acc is: {np.mean(ka)} \n')
    outputfile.write(f'The precision is: {prec}, cross-subject precision is: {np.mean(prec)} \n')
    outputfile.write(f'The recall is: {acc}, cross-subject recall is: {np.mean(recall)} \n')
    outputfile.write(f'The roc is: {roc}, cross-subject roc is: {np.mean(roc)} \n')
    outputfile.close()
    # outfile.close()

