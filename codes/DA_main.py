import random
import os
import sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_preprocessing.data_preprocessor import input_data_generation
from data_preprocessing.data_loader import get_loaders_with_domain
from sklearn.model_selection import LeaveOneOut
from models.model_factory import prepare_training, train_model_with_domain, test_evaluate
# sys.path.append(os.path.dirname(sys.path[0]))
# a = sys.path[0]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_root = 'models'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 250
# d_model = 50
# n_heads = 10
# 全卷积特征给你提取
d_model = 240
emb_feature = 30
n_heads = 12
d_ff = 32
n_layers = 1
n_epoch = 300
# lr = 0.01
dropout = 0.4
alpha_scala = 1

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
model_name = 'Mymodel_concat'
type = 'all'
from torchsummary import summary

# 打印日志设定
outputfile = open(f'logs/Mymodel/exp_{model_name}_dropout:{dropout}_attdim:{d_model}_encoder_num:{n_layers}_lr:{lr}.txt', 'w')
sys.stdout = outputfile

dataidx = [i for i in range(1, 10)]
loo = LeaveOneOut()
acc_ls = []
ka_ls = []
prec_ls = []
recall_ls = []
roc_auc_ls = []
for i, (train_idx, test_idx) in enumerate(loo.split(dataidx)):
    print(f"---------Start Fold {i} process---------")
    print(f'train index: {train_idx}, test index: {test_idx}')
    X_train, y_trian, y_train_domain, X_test, y_test, y_test_domain = input_data_generation(
        train_idx, test_idx)
    dataloaders = get_loaders_with_domain(X_train, y_trian, y_train_domain, X_test, y_test, y_test_domain, batch_size)
    train_sample = dataloaders['train'].dataset.X.shape[0]
    test_sample = dataloaders['test'].dataset.X.shape[0]
    dataset = {'dataset_sizes': {'train': train_sample, 'test': test_sample}}
    model, optimizer, lr_scheduler, criterion, device, criterion_domain = prepare_training(d_model, d_ff, n_heads, n_layers, dropout,
                                                                                           lr, model_name, type, emb_feature)
    print(summary(model, input_size=[(X_train.shape[1], X_train.shape[2]),(1,alpha_scala)]))
    best_model = train_model_with_domain(model, criterion, criterion_domain, optimizer, lr_scheduler, device, dataloaders, n_epoch, dataset)
    acc, ka, prec, recall, roc_auc = test_evaluate(best_model, device, X_test, y_test, model_name)
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








