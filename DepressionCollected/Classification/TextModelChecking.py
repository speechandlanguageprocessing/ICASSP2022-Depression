
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import wave
import re
import os
import tensorflow.compat.v1 as tf
import random
import itertools

from sklearn.preprocessing import StandardScaler
import pickle

# prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
# text_features = np.load(os.path.join(prefix, 'Features/Text/whole_samples_clf_avg.npz'))['arr_0']
# text_targets = np.load(os.path.join(prefix, 'Features/Text/whole_labels_clf_avg.npz'))['arr_0']

# audio_dep_idxs = np.where(text_targets == 1)[0]
# audio_non_idxs = np.where(text_targets == 0)[0]
# # train_dep_idxs_tmp = np.load(os.path.join(prefix, 'Features/Text/train_dep_idxs_0.80.npy'), allow_pickle=True)
# # train_non_idxs = list(np.load(os.path.join(prefix, 'Features/Text/train_non_idxs_0.80.npy'), allow_pickle=True))
# # train_dep_idxs_tmp = np.load(os.path.join(prefix, 'Features/Text/train_dep_idxs_0.65_2.npy'), allow_pickle=True)
# # train_non_idxs = list(np.load(os.path.join(prefix, 'Features/Text/train_non_idxs_0.65_2.npy'), allow_pickle=True))
# train_dep_idxs_tmp = np.load(os.path.join(prefix, 'Features/Text/train_dep_idxs_0.89_3.npy'), allow_pickle=True)
# train_non_idxs = list(np.load(os.path.join(prefix, 'Features/Text/train_non_idxs_0.89_3.npy'), allow_pickle=True))

# test_dep_idxs_tmp = list(set(audio_dep_idxs) - set(train_dep_idxs_tmp))
# test_non_idxs = list(set(audio_non_idxs) - set(train_non_idxs))

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
text_features = np.load(os.path.join(
    prefix, 'Features/TextWhole/whole_samples_clf_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(
    prefix, 'Features/TextWhole/whole_labels_clf_avg.npz'))['arr_0']
text_dep_idxs_tmp = np.where(text_targets == 1)[0]
text_non_idxs = np.where(text_targets == 0)[0]




# # training data augmentation
# train_dep_idxs = []
# for idx in train_dep_idxs_tmp:
#     feat = text_features[idx]
#     for i in itertools.permutations(feat, feat.shape[0]):
#         text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
#         text_targets = np.hstack((text_targets, 1))
#         train_dep_idxs.append(len(text_features)-1)

#         text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
#         text_targets = np.hstack((text_targets, 1))
#         train_dep_idxs.append(len(text_features)-1)

# # test data augmentation
# test_dep_idxs = []
# for idx in test_dep_idxs_tmp:
#     feat = text_features[idx]
#     for i in itertools.permutations(feat, feat.shape[0]):
#         text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
#         text_targets = np.hstack((text_targets, 1))
#         test_dep_idxs.append(len(text_features)-1)

def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])


def model_performance(y_test, y_test_pred_proba):
    """
    Evaluation metrics for network performance.
    """
    # y_test_pred = y_test_pred_proba.data.max(1, keepdim=True)[1]
    y_test_pred = y_test_pred_proba

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_test_pred, conf_matrix


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
            # nn.ReLU(),
            nn.Softmax(dim=1),
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

class BiLSTM(nn.Module):
    def __init__(self, rnn_layers, dropout, num_classes, text_hidden_dims, text_embed_size):
        super(BiLSTM, self).__init__()

        self.text_embed_size = text_embed_size
        self.text_hidden_dims = text_hidden_dims
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.num_classes = num_classes

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
            nn.Dropout(self.dropout),
            nn.Linear(self.text_hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),
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

    def forward(self, x_text):
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

        return text_feature

def evaluate(model, test_idxs):
    model.eval()
    batch_idx = 1
    total_loss = 0
    pred = torch.empty(config['batch_size'], 1).type(torch.LongTensor)
    # X_test = text_features[test_dep_idxs+test_non_idxs]
    # Y_test = text_targets[test_dep_idxs+test_non_idxs]
    X_test = text_features[test_idxs]
    Y_test = text_targets[test_idxs]
    global max_train_acc, max_acc, max_f1
    for i in range(0, X_test.shape[0], config['batch_size']):
        if i + config['batch_size'] > X_test.shape[0]:
            x, y = X_test[i:], Y_test[i:]
        else:
            x, y = X_test[i:(i+config['batch_size'])
                          ], Y_test[i:(i+config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(
            ),             Variable(torch.from_numpy(y)).cuda()
        else:
            x, y = Variable(torch.from_numpy(x).type(
                torch.FloatTensor), requires_grad=True), Variable(torch.from_numpy(y))
        with torch.no_grad():
            output = model(x.squeeze(2))
        pred = torch.cat((pred, output.data.max(1, keepdim=True)[1]))

    y_test_pred, conf_matrix = model_performance(
        Y_test, pred[config['batch_size']:])
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] +
                     conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / \
        (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}\n".format(f1_score))
    print('='*89)
    return precision, recall, f1_score

text_model_paths = ['BiLSTM_128_0.64_1.pt', 'BiLSTM_128_0.66_2.pt', 'BiLSTM_128_0.66_3.pt']
train_idxs_tmps = [np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.63_1.npy'), allow_pickle=True),
                   np.load(os.path.join(
                       prefix, 'Features/TextWhole/train_idxs_0.60_2.npy'), allow_pickle=True),
                   np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.60_3.npy'), allow_pickle=True)]
resample_idxs = [0, 1, 2, 3, 4, 5]
fold = 1
ps, rs, fs = [], [], []
for idx_i, train_idxs_tmp in enumerate(train_idxs_tmps):
    test_idxs_tmp = list(
        set(list(text_dep_idxs_tmp)+list(text_non_idxs)) - set(train_idxs_tmp))
    train_idxs, test_idxs = [], []
    # depression data augmentation
    for idx in train_idxs_tmp:
        if idx in text_dep_idxs_tmp:
            feat = text_features[idx]
            count = 0
            for i in itertools.permutations(feat, feat.shape[0]):
                if count in resample_idxs:
                    text_features = np.vstack(
                        (text_features, np.expand_dims(list(i), 0)))
                    text_targets = np.hstack((text_targets, 1))
                    train_idxs.append(len(text_features)-1)
                count += 1
        else:
            train_idxs.append(idx)

    for idx in test_idxs_tmp:
        if idx in text_dep_idxs_tmp:
            feat = text_features[idx]
            count = 0
            # resample_idxs = random.sample(range(6), 4)
            resample_idxs = [0,1,4,5]
            for i in itertools.permutations(feat, feat.shape[0]):
                if count in resample_idxs:
                    text_features = np.vstack(
                        (text_features, np.expand_dims(list(i), 0)))
                    text_targets = np.hstack((text_targets, 1))
                    test_idxs.append(len(text_features)-1)
                count += 1
        else:
            test_idxs.append(idx)

    config = {
        'num_classes': 2,
        'dropout': 0.5,
        'rnn_layers': 2,
        'embedding_size': 1024,
        'batch_size': 4,
        'epochs': 100,
        'learning_rate': 2e-5,
        'hidden_dims': 128,
        'bidirectional': True,
        'cuda': False,
    }

    text_lstm_model = torch.load(os.path.join(
        prefix, 'Model/ClassificationWhole/Text/{}'.format(text_model_paths[idx_i])))

    model = BiLSTM(config['rnn_layers'], config['dropout'], config['num_classes'],
                   config['hidden_dims'], config['embedding_size'])

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
    # model_state_dict = text_lstm_model.state_dict()
    # model.load_state_dict(model_state_dict)

    # evaluate(text_features_test, fuse_targets_test, audio_lstm_model)
    # evaluate(model, test_idxs)
    
    p, r, f = evaluate(text_lstm_model, test_idxs)
    ps.append(p)
    rs.append(r)
    fs.append(f)
print('precison: {} \n recall: {} \n f1 score: {}'.format(np.mean(ps), np.mean(rs), np.mean(fs)))
