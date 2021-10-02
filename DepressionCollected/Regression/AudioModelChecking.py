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


prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))
audio_features = np.squeeze(np.load(os.path.join(prefix, 'Features/AudioWhole/whole_samples_reg_256.npz'))['arr_0'], axis=2)
audio_targets = np.load(os.path.join(prefix, 'Features/AudioWhole/whole_labels_reg_256.npz'))['arr_0']

audio_dep_idxs = np.where(audio_targets >= 53)[0]
audio_non_idxs = np.where(audio_targets < 53)[0]
dep_idxs = np.load(os.path.join(prefix, 'Features/AudioWhole/dep_idxs.npy'), allow_pickle=True)
non_idxs = np.load(os.path.join(prefix, 'Features/AudioWhole/non_idxs.npy'), allow_pickle=True)

config = {
    'num_classes': 1,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 256,
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 5e-5,
    'hidden_dims': 256,
    'bidirectional': False,
    'cuda': False
}

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

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
 
def evaluate(fold, model):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global min_mae, min_rmse, test_dep_idxs, test_non_idxs
    pred = np.array([])
    X_test = audio_features[list(test_dep_idxs)+list(test_non_idxs)]
    Y_test = audio_targets[list(test_dep_idxs)+list(test_non_idxs)]
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
fold = 2
audio_lstm_model = torch.load(os.path.join(prefix, 'Model/Regression/Audio%d/gru_vlad256_256_8.25.pt'%(fold+1)))
model = AudioBiLSTM(config)
# model_state_dict = {}
# model_state_dict['lstm_net_audio.weight_ih_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_ih_l0']
# model_state_dict['lstm_net_audio.weight_hh_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_hh_l0']
# model_state_dict['lstm_net_audio.bias_ih_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_ih_l0']
# model_state_dict['lstm_net_audio.bias_hh_l0'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_hh_l0']

# model_state_dict['lstm_net_audio.weight_ih_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_ih_l1']
# model_state_dict['lstm_net_audio.weight_hh_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.weight_hh_l1']
# model_state_dict['lstm_net_audio.bias_ih_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_ih_l1']
# model_state_dict['lstm_net_audio.bias_hh_l1'] = audio_lstm_model.state_dict()['lstm_net_audio.bias_hh_l1']

# model_state_dict['fc_audio.1.weight'] = audio_lstm_model.state_dict()['fc_audio.1.weight']
# model_state_dict['fc_audio.1.bias'] = audio_lstm_model.state_dict()['fc_audio.1.bias']
# model_state_dict['fc_audio.4.weight'] = audio_lstm_model.state_dict()['fc_audio.4.weight']
# model_state_dict['fc_audio.4.bias'] = audio_lstm_model.state_dict()['fc_audio.4.bias']
model_state_dict = audio_lstm_model.state_dict()
model.load_state_dict(model_state_dict, strict=True)

test_dep_idxs_tmp = dep_idxs[fold*10:(fold+1)*10]
test_non_idxs = non_idxs[fold*44:(fold+1)*44]
train_dep_idxs_tmp = list(set(dep_idxs) - set(test_dep_idxs_tmp))
train_non_idxs = list(set(non_idxs) - set(test_non_idxs))

# training data augmentation
train_dep_idxs = []
for (i, idx) in enumerate(train_dep_idxs_tmp):
    feat = audio_features[idx]
    if i < 14:
        for i in itertools.permutations(feat, feat.shape[0]):
            audio_features = np.vstack((audio_features, np.expand_dims(list(i), 0)))
            audio_targets = np.hstack((audio_targets, audio_targets[idx]))
            train_dep_idxs.append(len(audio_features)-1)
    else:
        train_dep_idxs.append(idx)

# test data augmentation
# test_dep_idxs = []
# for idx in test_dep_idxs_tmp:
#     feat = audio_features[idx]
#     for i in itertools.permutations(feat, feat.shape[0]):
#         audio_features = np.vstack((audio_features, np.expand_dims(list(i), 0)))
#         audio_targets = np.hstack((audio_targets, audio_targets[idx]))
#         test_dep_idxs.append(len(audio_features)-1)
test_dep_idxs = test_dep_idxs_tmp

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.SmoothL1Loss()
# criterion = FocalLoss(class_num=2)
# evaluate(fold, model)
evaluate(fold, model)
