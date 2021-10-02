import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os
import pickle
import random
import itertools

prefix = os.path.abspath(os.path.join(os.getcwd(), "../"))
text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_reg_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_reg_avg.npz'))['arr_0']

dep_idxs = np.load(os.path.join(prefix, 'Features/AudioWhole/dep_idxs.npy'), allow_pickle=True)
non_idxs = np.load(os.path.join(prefix, 'Features/AudioWhole/non_idxs.npy'), allow_pickle=True)

config = {
    'num_classes': 1,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 1024,
    'batch_size': 2,
    'epochs': 110,
    'learning_rate': 1e-5,
    'hidden_dims': 128,
    'bidirectional': True,
    'cuda': False,
}

class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']

        self.build_model()
        self.init_weight()
        
    def init_weight(net):
        for name, param in net.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)
        
        # self.init_weight()
        
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            nn.ReLU(),
            # nn.Softmax(dim=1),
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        
        # x : [len_seq, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc_out(atten_out)

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
 
def train(epoch):
    global lr, train_acc
    model.train()
    batch_idx = 1      
    total_loss = 0
    correct = 0
    pred = np.array([])
    X_train = text_features[train_dep_idxs+train_non_idxs]
    Y_train = text_targets[train_dep_idxs+train_non_idxs]
    for i in range(0, X_train.shape[0], config['batch_size']):
        if i + config['batch_size'] > X_train.shape[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i + config['batch_size'])], Y_train[i:(
                i + config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        else:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), \
                Variable(torch.from_numpy(y)).type(torch.FloatTensor)

        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y.view_as(output))
        # 后向传播调整参数
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        batch_idx += 1
        # loss.item()能够得到张量中的元素值
        pred = np.hstack((pred, output.flatten().detach().numpy()))
        total_loss += loss.item()
    train_mae = mean_absolute_error(Y_train, pred)

    print('Train Epoch: {:2d}\t Learning rate: {:.4f}\t Loss: {:.4f}\t MAE: {:.4f}\t RMSE: {:.4f}\n '
        .format(epoch + 1, config['learning_rate'], total_loss, train_mae, \
            np.sqrt(mean_squared_error(Y_train, pred))))
    return train_mae


def evaluate(fold, model, train_mae):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global min_mae, min_rmse, test_dep_idxs, test_non_idxs
    pred = np.array([])
    X_test = text_features[list(test_dep_idxs)+list(test_non_idxs)]
    Y_test = text_targets[list(test_dep_idxs)+list(test_non_idxs)]
    with torch.no_grad():
        if config['cuda']:
            x, y = Variable(torch.from_numpy(X_test).type(torch.FloatTensor), requires_grad=True).cuda(),\
                Variable(torch.from_numpy(Y_test)).cuda()
        else:
            x, y = Variable(torch.from_numpy(X_test).type(torch.FloatTensor), requires_grad=True), \
                Variable(torch.from_numpy(Y_test)).type(torch.FloatTensor)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y.view_as(output))
        total_loss += loss.item()
        pred = output.flatten().detach().numpy()

        mae = mean_absolute_error(Y_test, pred)
        rmse = np.sqrt(mean_squared_error(Y_test, pred))

        print('MAE: {:.4f}\t RMSE: {:.4f}\n'.format(mae, rmse))
        print('='*89)

        if mae <= min_mae and mae < 8.5 and train_mae < 13:
            min_mae = mae
            min_rmse = rmse
            mode = 'bi' if config['bidirectional'] else 'norm'
            mode ='gru'
            save(model, os.path.join(prefix, 'Model/Regression/Text{}/BiLSTM_{}_{:.2f}'.format(fold+1, config['hidden_dims'], min_mae)))
            print('*' * 64)
            print('model saved: mae: {}\t rmse: {}'.format(min_mae, min_rmse))
            print('*' * 64)

    return total_loss

for fold in range(3):
    test_dep_idxs_tmp = dep_idxs[fold*10:(fold+1)*10]
    test_non_idxs = non_idxs[fold*44:(fold+1)*44]
    train_dep_idxs_tmp = list(set(dep_idxs) - set(test_dep_idxs_tmp))
    train_non_idxs = list(set(non_idxs) - set(test_non_idxs))

    # training data augmentation
    train_dep_idxs = []
    for (i, idx) in enumerate(train_dep_idxs_tmp):
        feat = text_features[idx]
        if i < 14:
            for i in itertools.permutations(feat, feat.shape[0]):
                text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
                text_targets = np.hstack((text_targets, text_targets[idx]))
                train_dep_idxs.append(len(text_features)-1)
        else:
            train_dep_idxs.append(idx)

    # test data augmentation
    # test_dep_idxs = []
    # for idx in test_dep_idxs_tmp:
    #     feat = text_features[idx]
    #     for i in itertools.permutations(feat, feat.shape[0]):
    #         text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
    #         text_targets = np.hstack((text_targets, text_targets[idx]))
    #         test_dep_idxs.append(len(text_features)-1)
    test_dep_idxs = test_dep_idxs_tmp


    model = TextBiLSTM(config)

    if config['cuda']:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.SmoothL1Loss()
    # criterion = FocalLoss(class_num=2)
    min_mae = 100
    min_rmse = 100
    train_mae = 100


    for ep in range(1, config['epochs']):
        train_mae = train(ep)
        tloss = evaluate(fold, model, train_mae)

# ============== prep ==============
# X_test = np.squeeze(np.load(os.path.join(prefix, 'Features/Audio/val_samples_reg_avid256.npz'))['arr_0'], axis=2)
# Y_test = np.load(os.path.join(prefix, 'Features/Audio/val_labels_reg_avid256.npz'))['arr_0']
# ============== prep ==============


# ============== SVM ==============

# from sklearn.svm import SVR
# from sklearn.model_selection import KFold

# X = text_features[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# Y = text_targets[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# kf = KFold(n_splits=3)
# regr = SVR(kernel='linear', gamma='auto')
# maes, rmses = [], []
# for train_index, test_index in kf.split(X):
#     # X_train, X_test = X[train_index], X[test_index]
#     # Y_train, Y_test = Y[train_index], Y[test_index]
#     X_train, Y_train = X[train_index], Y[train_index]
#     regr.fit([f.flatten() for f in X_train], Y_train)
#     pred = regr.predict([f.flatten() for f in X_test])

#     mae = mean_absolute_error(Y_test, pred)
#     rmse = np.sqrt(mean_squared_error(Y_test, pred))
#     maes.append(mae)
#     rmses.append(rmse)

#     print('MAE: {:.4f}\t RMSE: {:.4f}\n'.format(mae, rmse))
#     print('='*89)
#     # break

# print(np.mean(maes), np.mean(rmses))
# ============== SVM ==============

# # ============== DT ==============
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import KFold

# X = text_features[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# Y = text_targets[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# kf = KFold(n_splits=3)
# regr = DecisionTreeRegressor(max_depth=100, random_state=0, criterion="mse")
# maes, rmses = [], []
# for train_index, test_index in kf.split(X):
#     # X_train, X_test = X[train_index], X[test_index]
#     # Y_train, Y_test = Y[train_index], Y[test_index]
#     X_train, Y_train = X[train_index], Y[train_index]
#     regr.fit([f.flatten() for f in X_train], Y_train)
#     pred = regr.predict([f.flatten() for f in X_test])

#     mae = mean_absolute_error(Y_test, pred)
#     rmse = np.sqrt(mean_squared_error(Y_test, pred))
#     maes.append(mae)
#     rmses.append(rmse)

#     print('MAE: {:.4f}\t RMSE: {:.4f}\n'.format(mae, rmse))
#     print('='*89)

# print(np.mean(maes), np.mean(rmses))
# # ============== DT ==============

# # ============== RF ==============
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold

# X = text_features[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# Y = text_targets[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# kf = KFold(n_splits=3)
# regr = RandomForestRegressor(max_depth=100, random_state=0, criterion="mse")
# maes, rmses = [], []
# for train_index, test_index in kf.split(X):
#     # X_train, X_test = X[train_index], X[test_index]
#     # Y_train, Y_test = Y[train_index], Y[test_index]
#     X_train, Y_train = X[train_index], Y[train_index]
#     regr.fit([f.flatten() for f in X_train], Y_train)
#     pred = regr.predict([f.flatten() for f in X_test])

#     mae = mean_absolute_error(Y_test, pred)
#     rmse = np.sqrt(mean_squared_error(Y_test, pred))
#     maes.append(mae)
#     rmses.append(rmse)

#     print('MAE: {:.4f}\t RMSE: {:.4f}\n'.format(mae, rmse))
#     print('='*89)

# print(np.mean(maes), np.mean(rmses))
# # ============== RF ==============

# ============== ada ==============
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.model_selection import KFold

# X = text_features[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# Y = text_targets[train_dep_idxs+train_non_idxs+test_dep_idxs+test_non_idxs]
# kf = KFold(n_splits=3)
# regr = AdaBoostRegressor(n_estimators=50)
# maes, rmses = [], []
# for train_index, test_index in kf.split(X):
#     # X_train, X_test = X[train_index], X[test_index]
#     # Y_train, Y_test = Y[train_index], Y[test_index]
#     X_train, Y_train = X[train_index], Y[train_index]
#     regr.fit([f.flatten() for f in X_train], Y_train)
#     pred = regr.predict([f.flatten() for f in X_test])

#     mae = mean_absolute_error(Y_test, pred)
#     rmse = np.sqrt(mean_squared_error(Y_test, pred))
#     maes.append(mae)
#     rmses.append(rmse)

#     print('MAE: {:.4f}\t RMSE: {:.4f}\n'.format(mae, rmse))
#     print('='*89)

# print(np.mean(maes), np.mean(rmses))
# ============== ada ==============
