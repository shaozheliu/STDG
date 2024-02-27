import torch
import torch.nn as nn
from models.functions import ReverseLayerF
from models.MyTransformer import ST_Transformer_feature_extractor,CNN_Embedding, CNN_Embedding_Deepconv, \
    ST_Transformer_feature_extractor_sec, CNN_Embedding_Deepconv_2b, ST_Transformer_feature_extractor_2b

class Mymodel(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, type = 'all', clf_class=4, domain_class=8):
        super(Mymodel, self).__init__()

        ## Embedding layer ##
        self.embedding = nn.Sequential()
        # self.embedding.add_module('CNN_embedding',CNN_Embedding())
        self.embedding.add_module('CNN_embedding',CNN_Embedding_Deepconv(22,dropout))

        ## Feature extraction layer
        self.feature = nn.Sequential()
        self.feature.add_module('Transformer extractor',ST_Transformer_feature_extractor_sec(d_model, d_ff, n_heads, n_layers, dropout, type))

        # Class classifier layer
        self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(250, 32))
        self.class_classifier.add_module('c_fc1', nn.Linear(400, 32))
        # self.class_classifier.add_module('c_fc1', nn.Linear(4840, 32))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(32))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(dropout))
        self.class_classifier.add_module('c_fc2', nn.Linear(32, clf_class))

        # Domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(400, 32))
        # self.domain_classifier.add_module('d_fc1', nn.Linear(4840, 32))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(32))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_drop1', nn.Dropout(dropout))
        self.domain_classifier.add_module('d_fc2', nn.Linear(32, domain_class))

    def forward(self, x, alpha):
        x = self.embedding(x)  # （2，5，50）
        feature = self.feature(x)
        # feature = feature.view(-1,50*4*4)   # 将维度拉平
        feature = x + feature   # 残差
        feature = feature.reshape(-1,feature.shape[1] * feature.shape[2])
        # feature = feature.view(-1,feature.shape[1] * feature.shape[2])  # 将维度拉平
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output


class Mymodel_concat(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, emb_feature, type = 'all', clf_class=4, domain_class=8):
        super(Mymodel_concat, self).__init__()

        ## Embedding layer ##
        self.embedding = nn.Sequential()
        # self.embedding.add_module('CNN_embedding',CNN_Embedding())
        self.embedding.add_module('CNN_embedding',CNN_Embedding_Deepconv(22, emb_feature, dropout))

        ## Feature extraction layer
        self.feature = nn.Sequential()
        self.feature.add_module('Transformer extractor',ST_Transformer_feature_extractor_sec(d_model, d_ff, n_heads, n_layers, dropout,emb_feature, type))

        ## method1
        # Class classifier layer
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc2', nn.Linear(900, clf_class))
        #
        # # Domain classifier
        # self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc2', nn.Linear(900, domain_class))

        ## method2
        # Class classifier layer
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(240, clf_class))

        # Domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(240, domain_class))


        # Class classifier layer
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(250, 32))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # # self.class_classifier.add_module('c_fc1', nn.Linear(240, 64))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(32))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout(dropout))
        # self.class_classifier.add_module('c_fc2', nn.Linear(900, clf_class))

        # Domain classifier
        # self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc1', nn.Linear(240, 32))
        # # self.domain_classifier.add_module('d_fc1', nn.Linear(4840, 32))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(32))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_drop1', nn.Dropout(dropout))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(900, domain_class))

    def forward(self, x, alpha):
        x = self.embedding(x)  # （2，5，50）
        feature = self.feature(x)
        # feature = x + feature   # 残差
        # feature = feature.view(-1,feature.shape[1] * feature.shape[2])  # 将维度拉平
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output


class Mymodel_2b(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, emb_feature, type = 'all', clf_class=2, domain_class=8):
        super(Mymodel_2b, self).__init__()

        ## Embedding layer ##
        self.embedding = nn.Sequential()
        # self.embedding.add_module('CNN_embedding',CNN_Embedding())
        self.embedding.add_module('CNN_embedding',CNN_Embedding_Deepconv_2b(3, emb_feature, dropout))

        ## Feature extraction layer
        self.feature = nn.Sequential()
        self.feature.add_module('Transformer extractor',ST_Transformer_feature_extractor_2b(d_model, d_ff, n_heads, n_layers, dropout,emb_feature, type))

        ## method1
        # Class classifier layer
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc2', nn.Linear(900, clf_class))
        #
        # # Domain classifier
        # self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc2', nn.Linear(900, domain_class))

        ## method2
        # Class classifier layer
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(930, clf_class))

        # Domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(930, domain_class))


        # Class classifier layer
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(250, 32))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # # self.class_classifier.add_module('c_fc1', nn.Linear(240, 64))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(32))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout(dropout))
        # self.class_classifier.add_module('c_fc2', nn.Linear(900, clf_class))

        # Domain classifier
        # self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc1', nn.Linear(240, 32))
        # # self.domain_classifier.add_module('d_fc1', nn.Linear(4840, 32))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(32))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_drop1', nn.Dropout(dropout))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(900, domain_class))

    def forward(self, x, alpha):
        x = self.embedding(x)  # （2，30，250）
        feature = self.feature(x)
        # feature = x + feature   # 残差
        # feature = feature.view(-1,feature.shape[1] * feature.shape[2])  # 将维度拉平
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output




class Mymodel_ablation(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, clf_class=4, domain_class=8):
        super(Mymodel_ablation, self).__init__()

        ## Embedding layer ##
        self.embedding = nn.Sequential()
        self.embedding.add_module('CNN_embedding',CNN_Embedding())
        # self.embedding.add_module('CNN_embedding',CNN_Embedding_Deepconv(22,dropout))

        ## Feature extraction layer
        self.feature = nn.Sequential()
        self.feature.add_module('Transformer extractor',ST_Transformer_feature_extractor(d_model, d_ff, n_heads, n_layers, dropout))

        # Class classifier layer
        self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(250, 32))
        self.class_classifier.add_module('c_fc1', nn.Linear(240, 32))
        # self.class_classifier.add_module('c_fc1', nn.Linear(4840, 32))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(32))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(dropout))
        self.class_classifier.add_module('c_fc2', nn.Linear(32, clf_class))


    def forward(self, x):
        x = self.embedding(x)  # （2，5，50）
        feature = self.feature(x)
        # feature = feature.view(-1,50*4*4)   # 将维度拉平
        feature = x + feature * 0.4  # 残差
        feature = feature.reshape(-1,feature.shape[1] * feature.shape[2])
        # feature = feature.view(-1,feature.shape[1] * feature.shape[2])  # 将维度拉平
        class_output = self.class_classifier(feature)
        return class_output


if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 22, 250))
    head = 5
    d_model = 70
    d_ff = 512
    ff_hide = 1024
    mode1 = "T"
    mode2 = "C"
    n_layer = 3
    dropout = 0.1
    model = Mymodel(d_model, d_ff, head, n_layer, dropout)
    out  = model(inp)