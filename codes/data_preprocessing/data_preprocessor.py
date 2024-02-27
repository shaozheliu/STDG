import h5py
import numpy as np
from sklearn.model_selection import LeaveOneOut

def import_data(every=False):
    if every:
        electrodes = 25
    else:
        electrodes = 22
    X, y = [], []
    for i in range(9):
        A01T = h5py.File('/home/alk/Data/BCI2aIV-mat/A0' + str(i + 1) + 'T_slice.mat', 'r')
        X1 = np.copy(A01T['image'])
        X.append(X1[:, :electrodes, :])
        y1 = np.copy(A01T['type'])
        y1 = y1[0, 0:X1.shape[0]:1]
        y.append(np.asarray(y1, dtype=np.int32))

    for subject in range(9):
        delete_list = []
        for trial in range(288):
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    y = [y[i] - np.min(y[i]) for i in range(len(y))]
    return X, y

def import_subjecti_data(every=False, sub_list=[0,1,2,3]):
    if every:
        electrodes = 25
    else:
        electrodes = 22
    X, y = {}, {}
    # X, y = [], []
    for i in sub_list:
        A01T = h5py.File('/home/alk/Data/BCI2aIV-mat/A0' + str(i+1) + 'T_slice.mat', 'r')
        X1 = np.copy(A01T['image'])
        # X.append(X1[:, :electrodes, :])
        X[i] = X1[:, :electrodes, :]
        y1 = np.copy(A01T['type'])
        y1 = y1[0, 0:X1.shape[0]:1]
        # y.append(np.asarray(y1, dtype=np.int32))
        y[i] = np.asarray(y1, dtype=np.int32)

    for subject in sub_list:
        delete_list = []
        for trial in range(288):
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    # 将dict元素写回矩阵
    X_mat, y_mat = [], []
    for i in X:
        X_mat.append(X[i])
    for j in y:
        y_mat.append(y[j])
    y_mat = [y_mat[i] - np.min(y_mat[i]) for i in range(len(y_mat))]
    return X_mat, y_mat



def import_data_test(every=False):
    if every:
        electrodes = 25
    else:
        electrodes = 22
    X, y = [],[]
    for i in range(9):
        B01T = h5py.File('grazdata/B0' + str(i + 1) + 'T.mat', 'r')
        X1 = np.copy(B01T['image'])
        X.append(X1[:, :electrodes, :])
        y1 = np.copy(B01T['type'])
        y1 = y1[0, 0:X1.shape[0]:1]
        y.append(np.asarray(y1, dtype=np.int32))

    for subject in range(9):
        delete_list = []
        for trial in range(288):
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    y = [y[i] - np.min(y[i]) for i in range(len(y))]
    return X, y

def train_test_subject(X, y, train_all=True, standardize=True):

    l = np.random.permutation(len(X[0]))
    X_test = X[0][l[:50], :, :]
    y_test = y[0][l[:50]]

    if train_all:
        X_train = np.concatenate((X[0][l[50:], :, :], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
        y_train = np.concatenate((y[0][l[50:]], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    else:
        X_train = X[0][l[50:], :, :]
        y_train = y[0][l[50:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test


def train_test_total(X, y, standardize=True, train_val_test_ratio= [0.6,0.2,0.2]):

    X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
    y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))
    train_idx = int(train_val_test_ratio[0] * X_total.shape[0])
    val_idx = int(train_val_test_ratio[1] * X_total.shape[0])
    test_idx = X_total.shape[0] - train_idx - val_idx
    l = np.random.permutation(len(X_total))
    # X_valid = X_total[l[:200], :, :]
    # y_valid = y_total[l[:200]]
    # X_test = X_total[l[200:400], :, :]
    # y_test = y_total[l[200:400]]
    # X_train = X_total[l[400:], :, :]
    # y_train = y_total[l[400:]]
    X_train = X_total[l[:train_idx], :, :]
    y_train = y_total[l[:train_idx]]
    X_valid = X_total[l[train_idx:train_idx+val_idx], :, :]
    y_valid = y_total[l[train_idx:train_idx+val_idx]]
    X_test = X_total[l[train_idx+val_idx:], :, :]
    y_test = y_total[l[train_idx+val_idx:]]


    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var
        X_valid -= X_train_mean
        X_valid /= X_train_var

    # X_train = np.transpose(X_train, (0, 2, 1))
    # X_test = np.transpose(X_test, (0, 2, 1))
    # X_valid = np.transpose(X_valid, (0, 2, 1))
    y_train = np.expand_dims(y_train,axis = -1)
    y_valid = np.expand_dims(y_valid, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    return X_train,X_valid, X_test, y_train, y_valid, y_test

##--------------------数据增广---------------------##
def data_argumentation(X, y):
    retX = []
    rety = []
    for (data,label) in zip(X,y):
        argument1 = data[:,:,:250]
        argument2 = data[:,:,250:500]
        argument3 = data[:,:,500:750]
        argument4 = data[:,:,750:1000]
        retX.append(np.concatenate((argument1,argument2,argument3,argument4),axis=0))
        rety.append(np.concatenate((label,label,label,label),axis=None))
    print("Finish data argumentation")
    return retX, rety

def data_argumentation_v2(X,y):
    retX = []
    rety = []
    for (data, label) in zip(X, y):
        argument1 = data[:, :, :500]
        argument2 = data[:, :, 500:]
        retX.append(np.concatenate((argument1, argument2), axis=0))
        rety.append(np.concatenate((label, label), axis=None))
    print("Finish data argumentation")
    return retX, rety

##-----------------domain label generator-----------------##
def domain_label_generation(X):
    domain_labels = []
    for i,subject in enumerate(X):
        domain_label = np.ones(subject.shape[0], dtype=int) * i # 这个dtype可能有问题
        domain_labels.append(domain_label)
    return X, domain_labels

def train_val_split_with_domain(X, y_label, y_domain_label, split_ratio):
    def get_idx():
        np.random.seed(seed=1024)
        # rng = np.random.choice(len(y), len(y), replace=False)
        rng = np.random.choice(y_label.shape[0], y_label.shape[0], replace=False)
        return rng
    indices = get_idx()
    train_idx, val_idx = indices[0: int(X.shape[0]*split_ratio)], \
                         indices[int(X.shape[0]*split_ratio): ]
    train_X, train_y, train_domain_y, val_X, val_y, val_domain_y = \
        X[train_idx], y_label[train_idx], y_domain_label[train_idx], \
        X[val_idx], y_label[val_idx], y_domain_label[val_idx]
    print("Finish split")
    return train_X, train_y, train_domain_y, val_X, val_y, val_domain_y



def input_data_generation(train_idx, test_idx):
    X, y = import_subjecti_data(False, train_idx)
    X, y_domain_label = domain_label_generation(X)
    X_test, y_test = import_subjecti_data(False, test_idx)
    X_test, y_test_domain = domain_label_generation(X_test)
    X = np.concatenate([X[i] for i in range(len(X))])
    y = np.concatenate([y[i] for i in range(len(y))])
    y_domain_label = np.concatenate([y_domain_label[i] for i in range(len(y_domain_label))])
    y_test_domain = np.concatenate([y_test_domain[i] for i in range(len(y_test_domain))])
    X_test = np.concatenate([X_test[i] for i in range(len(X_test))])
    y_test = np.concatenate([y_test[i] for i in range(len(y_test))])
    # X_train, y_trian, y_train_domain, X_valid, y_valid, y_valid_domain = train_val_split_with_domain(X, y, y_domain_label, 0.8)
    # return X_train, y_trian, y_train_domain, X_valid, y_valid, y_valid_domain, X_test, y_test, y_test_domain
    return X, y, y_domain_label, X_test, y_test, y_test_domain
if __name__ == "__main__":
    # X,y = import_data()
    # X,y = data_argumentation_v2(X,y)
    # X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_total(X,y)
    dataidx = [i for i in range(1, 10)]
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(dataidx):
        print(f'train index: {train_idx}, test index: {test_idx}')
        # X, y = import_subjecti_data(False,train_idx)
        # X, y_domain_label = domain_label_generation(X,y)
        # X_test, y_test = import_subjecti_data(False, test_idx)
        X_train, y_trian, y_train_domain, X_test, y_test, y_test_domain = input_data_generation(train_idx, test_idx)
        print(X_train.shape)
        print(X_test.shape)


