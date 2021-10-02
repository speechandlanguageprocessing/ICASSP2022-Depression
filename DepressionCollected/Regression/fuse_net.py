
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import wave
import librosa
from python_speech_features import *
import re
from allennlp.commands.elmo import ElmoEmbedder
import os
import tensorflow.compat.v1 as tf
import itertools

prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))

text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_reg_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_reg_avg.npz'))['arr_0']
audio_features = np.squeeze(np.load(os.path.join(prefix, 'Features/AudioWhole/whole_samples_reg_256.npz'))['arr_0'], axis=2)
audio_targets = np.load(os.path.join(prefix, 'Features/AudioWhole/whole_labels_reg_256.npz'))['arr_0']
fuse_features = [[audio_features[i], text_features[i]] for i in range(text_features.shape[0])]
fuse_targets = text_targets

fuse_dep_idxs = np.where(text_targets >= 53)[0]
fuse_non_idxs = np.where(text_targets < 53)[0]
dep_idxs = np.load(os.path.join(prefix, 'Features/AudioWhole/dep_idxs.npy'), allow_pickle=True)
non_idxs = np.load(os.path.join(prefix, 'Features/AudioWhole/non_idxs.npy'), allow_pickle=True)

text_model_paths = ['Model/Regression/Text1/BiLSTM_128_7.75.pt', 'Model/Regression/Text2/BiLSTM_128_8.46.pt', 'Model/Regression/Text3/BiLSTM_128_8.01.pt']
audio_model_paths = ['Model/Regression/Audio1/gru_vlad256_256_7.60.pt', 'Model/Regression/Audio2/gru_vlad256_256_8.38.pt', 'Model/Regression/Audio3/gru_vlad256_256_8.25.pt']

config = {
    'num_classes': 1,
    'dropout': 0.5,
    'rnn_layers': 2,
    'audio_embed_size': 256,
    'text_embed_size': 1024,
    'batch_size': 4,
    'epochs': 150,
    'learning_rate': 8e-5,
    'audio_hidden_dims': 256,
    'text_hidden_dims': 128,
    'cuda': False,
    'lambda': 1e-2,
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

class AudioBiLSTM(nn.Module):
    def __init__(self, config):
        super(AudioBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']

        self.build_model()

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
            nn.ReLU(inplace=True))
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        self.lstm_net_audio = nn.GRU(self.embedding_size,
                                self.hidden_dims,
                                num_layers=self.rnn_layers,
                                dropout=self.dropout,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        # self.lstm_net_audio = nn.GRU(self.embedding_size, self.hidden_dims,
        #                         num_layers=self.rnn_layers, dropout=self.dropout, batch_first=True)

        self.bn = nn.BatchNorm1d(3)

        # FC层
        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            nn.ReLU(),
            # nn.Softmax(dim=1)
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
        #         h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
       # print(atten_w.shape, m.transpose(1, 2).shape)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        x, _ = self.lstm_net_audio(x)
        # x = self.bn(x)
        x = x.sum(dim=1)
        out = self.fc_audio(x)
        return out

class fusion_net(nn.Module):
    def __init__(self, text_embed_size, text_hidden_dims, rnn_layers, dropout, num_classes, \
         audio_hidden_dims, audio_embed_size):
        super(fusion_net, self).__init__()
        self.text_embed_size = text_embed_size
        self.audio_embed_size = audio_embed_size
        self.text_hidden_dims = text_hidden_dims
        self.audio_hidden_dims = audio_hidden_dims
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.num_classes = num_classes
        
        # ============================= TextBiLSTM =================================
        
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.text_hidden_dims, self.text_hidden_dims),
            nn.ReLU(inplace=True)
        )

        # 双层lstm
        self.lstm_net = nn.LSTM(self.text_embed_size, self.text_hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=True)
        # FC层
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.text_hidden_dims, self.text_hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # ============================= TextBiLSTM =================================

        # ============================= AudioBiLSTM =============================

        self.lstm_net_audio = nn.GRU(self.audio_embed_size,
                                self.audio_hidden_dims,
                                num_layers=self.rnn_layers,
                                dropout=self.dropout,
                                bidirectional=False,
                                batch_first=True)

        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.audio_hidden_dims, self.audio_hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # ============================= AudioBiLSTM =============================

        # ============================= last fc layer =============================
        # self.bn = nn.BatchNorm1d(self.text_hidden_dims + self.audio_hidden_dims)
        # modal attention
        self.modal_attn = nn.Linear(self.text_hidden_dims + self.audio_hidden_dims, self.text_hidden_dims + self.audio_hidden_dims, bias=False)
        self.fc_final = nn.Sequential(
            nn.Linear(self.text_hidden_dims + self.audio_hidden_dims, self.num_classes, bias=False),
            nn.ReLU(),
            # nn.Softmax(dim=1),
            # nn.Sigmoid()
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
    
    def pretrained_feature(self, x):
        with torch.no_grad():
            x_text = []
            x_audio = []
            for ele in x:
                x_text.append(ele[1])
                x_audio.append(ele[0])
            x_text, x_audio = Variable(torch.tensor(x_text).type(torch.FloatTensor), requires_grad=False), Variable(torch.tensor(x_audio).type(torch.FloatTensor), requires_grad=False)
            # ============================= TextBiLSTM =================================
            # x : [len_seq, batch_size, embedding_dim]
            x_text = x_text.permute(1, 0, 2)
            output, (final_hidden_state, _) = self.lstm_net(x_text)
            # output : [batch_size, len_seq, n_hidden * 2]
            output = output.permute(1, 0, 2)
            # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
            final_hidden_state = final_hidden_state.permute(1, 0, 2)
            # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
            # atten_out = self.attention_net(output, final_hidden_state)
            atten_out = self.attention_net_with_w(output, final_hidden_state)
            text_feature = self.fc_out(atten_out)

            # ============================= TextBiLSTM =================================

            # ============================= AudioBiLSTM =============================

            x_audio, _ = self.lstm_net_audio(x_audio)
            x_audio = x_audio.sum(dim=1)
            audio_feature = self.fc_audio(x_audio)

        # ============================= AudioBiLSTM =============================
        return (text_feature, audio_feature)
        
    def forward(self, x): 
        # x = self.bn(x)
        modal_weights = torch.sigmoid(self.modal_attn(x))
        # modal_weights = self.modal_attn(x)
        x = (modal_weights * x)
        output = self.fc_final(x)
        return output
    
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        
    def forward(self, text_feature, audio_feature, target, model):
        weight = model.fc_final[0].weight
        # bias = model.fc_final[0].bias
        # print(weight, bias)
        pred_text = F.linear(text_feature, weight[:, :config['text_hidden_dims']])
        pred_audio = F.linear(audio_feature, weight[:, config['text_hidden_dims']:])
        # l = nn.CrossEntropyLoss()
        l = nn.SmoothL1Loss()
        target = torch.tensor(target).view_as(pred_text).float()
        return l(pred_text, target) + l(pred_audio, target)

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

def train(model, epoch):
    global max_train_acc, train_acc
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    pred = np.array([])
    X_train = []
    Y_train = []
    for idx in train_dep_idxs+train_non_idxs:
        X_train.append(fuse_features[idx])
        Y_train.append(fuse_targets[idx])
    for i in range(0, len(X_train), config['batch_size']):
        if i + config['batch_size'] > len(X_train):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+config['batch_size'])], Y_train[i:(i+config['batch_size'])]
        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        text_feature, audio_feature = model.pretrained_feature(x)
        audio_feature_norm = (audio_feature - audio_feature.mean())/audio_feature.std()
        text_feature_norm = (text_feature - text_feature.mean())/text_feature.std()
        # concat_x = torch.cat((text_feature_norm, audio_feature_norm), dim=1)
        concat_x = torch.cat((text_feature, audio_feature), dim=1)
        output = model(concat_x)
        # loss = criterion(output, torch.tensor(y).float())
        loss = criterion(text_feature, audio_feature, y, model)
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

def evaluate(model, fold, train_mae):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global min_mae, min_rmse, test_dep_idxs, test_non_idxs
    pred = np.array([])
    X_test = []
    Y_test = []
    for idx in list(test_dep_idxs)+list(test_non_idxs):
        X_test.append(fuse_features[idx])
        Y_test.append(fuse_targets[idx])
    for i in range(0, len(X_test), config['batch_size']):
        if i + config['batch_size'] > len(X_test):
            x, y = X_test[i:], Y_test[i:]
        else:
            x, y = X_test[i:(i+config['batch_size'])], Y_test[i:(i+config['batch_size'])]
        text_feature, audio_feature = model.pretrained_feature(x)
        with torch.no_grad():
            audio_feature_norm = (audio_feature - audio_feature.mean())/audio_feature.std()
            text_feature_norm = (text_feature - text_feature.mean())/text_feature.std()
            concat_x = torch.cat((text_feature, audio_feature), dim=1)
            # concat_x = torch.cat((text_feature_norm, audio_feature_norm), dim=1)
            output = model(concat_x)
        # loss = criterion(output, torch.tensor(y).float())
        loss = criterion(text_feature, audio_feature, y, model)
        pred = np.hstack((pred, output.flatten().detach().numpy()))
        total_loss += loss.item()
        
    mae = mean_absolute_error(Y_test, pred)
    rmse = np.sqrt(mean_squared_error(Y_test, pred))

    print('MAE: {:.4f}\t RMSE: {:.4f}\n'.format(mae, rmse))
    print('='*89)

    if mae <= min_mae and mae < 8.2 and train_mae < 13:
        min_mae = mae
        min_rmse = rmse
        save(model, os.path.join(prefix, 'Model/Regression/Fuse{}/fuse_{:.2f}'.format(fold+1, min_mae)))
        print('*' * 64)
        print('model saved: mae: {}\t rmse: {}'.format(min_mae, min_rmse))
        print('*' * 64)

    return total_loss

def evaluate_audio(model):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global min_mae, min_rmse, test_dep_idxs, test_non_idxs
    pred = np.array([])
    X_test = []
    Y_test = []
    for idx in list(test_dep_idxs)+list(test_non_idxs):
        X_test.append(fuse_features[idx][0])
        Y_test.append(fuse_targets[idx])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

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

def evaluate_text(model):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global min_mae, min_rmse, test_dep_idxs, test_non_idxs
    pred = np.array([])
    X_test = []
    Y_test = []
    for idx in list(test_dep_idxs)+list(test_non_idxs):
        X_test.append(fuse_features[idx][1])
        Y_test.append(fuse_targets[idx])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    criterion = nn.SmoothL1Loss()
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

for fold in range(3):
    test_dep_idxs_tmp = dep_idxs[fold*10:(fold+1)*10]
    test_non_idxs = non_idxs[fold*44:(fold+1)*44]
    train_dep_idxs_tmp = list(set(dep_idxs) - set(test_dep_idxs_tmp))
    train_non_idxs = list(set(non_idxs) - set(test_non_idxs))

    train_dep_idxs = []
    test_dep_idxs = []
    # depression data augmentation
    for (i, idx) in enumerate(train_dep_idxs_tmp):
        feat = fuse_features[idx]
        audio_perm = itertools.permutations(feat[0], 3)
        text_perm = itertools.permutations(feat[1], 3)
        if i < 14:
            for fuse_perm in zip(audio_perm, text_perm):
                fuse_features.append(fuse_perm)
                fuse_targets = np.hstack((fuse_targets, fuse_targets[idx]))
                train_dep_idxs.append(len(fuse_features)-1)
        else:
            train_dep_idxs.append(idx)

    test_dep_idxs = test_dep_idxs_tmp

    model = fusion_net(config['text_embed_size'], config['text_hidden_dims'], config['rnn_layers'], \
    config['dropout'], config['num_classes'], config['audio_hidden_dims'], config['audio_embed_size'])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.SmoothL1Loss()
    criterion = MyLoss()

    text_lstm_model = torch.load(os.path.join(prefix, text_model_paths[fold]))
    audio_lstm_model = torch.load(os.path.join(prefix, audio_model_paths[fold]))
    model_state_dict = {}
    model_state_dict['lstm_net_audio.weight_ih_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_ih_l0']
    model_state_dict['lstm_net_audio.weight_hh_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_hh_l0']
    model_state_dict['lstm_net_audio.bias_ih_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_ih_l0']
    model_state_dict['lstm_net_audio.bias_hh_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_hh_l0']

    model_state_dict['lstm_net_audio.weight_ih_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_ih_l1']
    model_state_dict['lstm_net_audio.weight_hh_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_hh_l1']
    model_state_dict['lstm_net_audio.bias_ih_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_ih_l1']
    model_state_dict['lstm_net_audio.bias_hh_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_hh_l1']

    model_state_dict['fc_audio.1.weight'] = audio_lstm_model.state_dict()['fc_audio.1.weight']
    model_state_dict['fc_audio.1.bias'] = audio_lstm_model.state_dict()['fc_audio.1.bias']
    model_state_dict['fc_audio.4.weight'] = audio_lstm_model.state_dict()['fc_audio.4.weight']
    model_state_dict['fc_audio.4.bias'] = audio_lstm_model.state_dict()['fc_audio.4.bias']
    model.load_state_dict(text_lstm_model.state_dict(), strict=False)
    # model.load_state_dict(audio_lstm_model.state_dict(), strict=False)
    model.load_state_dict(model_state_dict, strict=False)
    
    for param in model.parameters():
        param.requires_grad = True

    model.fc_final[0].weight.requires_grad = True
    # model.fc_final[0].bias.requires_grad = True
    model.modal_attn.weight.requires_grad = True
    min_mae = 100
    min_rmse = 100
    train_mae = 100

    for ep in range(1, config['epochs']):
        train_mae = train(model, ep)
        tloss = evaluate(model, fold, train_mae)
    # evaluate_audio(audio_lstm_model)
    # evaluate_text(text_lstm_model)