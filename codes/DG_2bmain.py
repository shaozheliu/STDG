import random
import os
import sys
import re
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_preprocessing.data_preprocessor import input_data_generation
from data_preprocessing.data_loader import get_loaders_with_domain
from data_preprocessing.data_processor_2b import DataProcess
from sklearn.model_selection import LeaveOneOut
from models.model_factory import prepare_training_2b, train_model_with_domain, test_evaluate, test_evaluate_2b\
    # , train_model_without_domain
# sys.path.append(os.path.dirname(sys.path[0]))
# a = sys.path[0]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data_2b_path = '/home/alk/Data/bci2b-npy/'
data_2b_files = ["B0101T","B0102T","B0103T",
					 "B0201T","B0202T","B0203T",
					 "B0301T","B0302T","B0303T",
					 "B0401T","B0402T","B0403T",
					 "B0501T","B0502T","B0503T",
					 "B0601T","B0602T","B0603T",
					 "B0701T","B0702T","B0703T",
					 "B0801T","B0802T","B0803T",
					 "B0901T","B0902T","B0903T"]

cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 250
# d_model = 50
# n_heads = 10
# 全卷积特征给你提取
d_model = 50
emb_feature = 50
n_heads = 5
d_ff = 12
n_layers = 1
n_epoch = 300
# lr = 0.01
dropout = 0.3
alpha_scala = 1

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
model_name = 'Mymodel_concat'
type = 'all'
domain_flg = True
from torchsummary import summary

# 打印日志设定
outputfile = open(f'logs/dataset2b/exp_{model_name}_dropout:{dropout}_attdim:{d_model}_encoder_num:{n_layers}_lr:{lr}.txt', 'w')
sys.stdout = outputfile

subject_list = ['01','02','03','04','05','06','07','08','09']
loo = LeaveOneOut()
acc_ls = []
ka_ls = []
prec_ls = []
recall_ls = []
roc_auc_ls = []
for i, (train_idx, test_idx) in enumerate(loo.split(subject_list)):
    print(f"---------Start Fold {i} process---------")
    print(f'train subject is: {train_idx}, test subject is: {test_idx}')
    train_subjects = []
    test_subjects = []
    for idx in train_idx:
        custfile1 = 'B0' + str(idx+1) + '01T'
        custfile2 = 'B0' + str(idx+1) + '02T'
        custfile3 = 'B0' + str(idx+1) + '03T'
        train_subjects.append(custfile1)
        train_subjects.append(custfile2)
        train_subjects.append(custfile3)
    for idx in test_idx:
        custfile1 = 'B0' + str(idx + 1) + '01T'
        custfile2 = 'B0' + str(idx + 1) + '02T'
        custfile3 = 'B0' + str(idx + 1) + '03T'
        test_subjects.append(custfile1)
        test_subjects.append(custfile2)
        test_subjects.append(custfile3)
    GetData_train = DataProcess(data_2b_path, data_2b_files, 2, train_subjects)
    X_train, y_train, y_train_domain = GetData_train.data, GetData_train.label, GetData_train.domain_label
    GetData_test = DataProcess(data_2b_path, data_2b_files, 2, test_subjects)
    X_test, y_test, y_test_domain = GetData_test.data, GetData_test.label, GetData_test.domain_label


    dataloaders = get_loaders_with_domain(X_train, y_train, y_train_domain, X_test, y_test, y_test_domain, batch_size)
    train_sample = dataloaders['train'].dataset.X.shape[0]
    test_sample = dataloaders['test'].dataset.X.shape[0]
    dataset = {'dataset_sizes': {'train': train_sample, 'test': test_sample}}
    model, optimizer, lr_scheduler, criterion, device, criterion_domain = prepare_training_2b(d_model, d_ff, n_heads, n_layers, dropout,
                                                                                           lr, model_name, type, emb_feature)
    print(summary(model, input_size=[(X_train.shape[1], X_train.shape[2]),(1,alpha_scala)]))
    if domain_flg == True:
        best_model = train_model_with_domain(model, criterion, criterion_domain, optimizer, lr_scheduler, device, dataloaders, n_epoch, dataset)
    else:
        best_model = train_model_without_domain(model, criterion, criterion_domain, optimizer, lr_scheduler, device,
                                         dataloaders, n_epoch, dataset)
    acc, ka, prec, recall, roc_auc = test_evaluate_2b(best_model, device, X_test, y_test, model_name)
    acc_ls.append(acc)
    ka_ls.append(ka)
    prec_ls.append(prec)
    recall_ls.append(recall)
    roc_auc_ls.append(roc_auc)

    print(f'The accuracy is: {acc_ls}, cross-subject acc is: {np.mean(acc_ls)} \n')
    print(f'The acohen_kappa_score is: {ka_ls}, cross-subject acohen_kappa_score is: {np.mean(ka_ls)} \n')
    print(f'The precision is: {prec_ls}, cross-subject precision is: {np.mean(prec_ls)} \n')
    print(f'The recall is: {recall_ls}, cross-subject recall is: {np.mean(recall_ls)} \n')
    print(f'The roc_auc is: {roc_auc_ls}, cross-subject roc is: {np.mean(roc_auc_ls)} \n')

outputfile.close()  # 关闭文件








