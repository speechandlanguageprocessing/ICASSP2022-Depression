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
from audio_gru_whole import AudioBiLSTM

from sklearn.preprocessing import StandardScaler
import pickle

class BiLSTM(nn.Module):
    def __init__(self, rnn_layers, dropout, num_classes, audio_hidden_dims, audio_embed_size):
        super(BiLSTM, self).__init__()

        self.lstm_net_audio = nn.GRU(audio_embed_size, audio_hidden_dims,
                                num_layers=rnn_layers, dropout=dropout, batch_first=True)

        self.fc_audio = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(audio_hidden_dims, audio_hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(audio_hidden_dims, num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x, _ = self.lstm_net_audio(x)
        # x = self.bn(x)
        x = x.sum(dim=1)
        out = self.fc_audio(x)
        return out

# prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
# audio_features = np.squeeze(np.load(os.path.join(prefix, 'Features/Audio/whole_samples_clf_avid256.npz'))['arr_0'], axis=2)
# audio_targets = np.load(os.path.join(prefix, 'Features/Audio/whole_labels_clf_avid256.npz'))['arr_0']

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
audio_features = np.squeeze(np.load(os.path.join(prefix, 'Features/AudioWhole/whole_samples_clf_256.npz'))['arr_0'], axis=2)
audio_targets = np.load(os.path.join(prefix, 'Features/AudioWhole/whole_labels_clf_256.npz'))['arr_0']

audio_dep_idxs = np.where(audio_targets == 1)[0]
audio_non_idxs = np.where(audio_targets == 0)[0]

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

config = {
    'num_classes': 2,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 256,
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 1e-5,
    'hidden_dims': 256,
    'bidirectional': False,
    'cuda': False
}

# audio_lstm_model = torch.load(os.path.join(prefix, 'Model/Classification/Audio/BiLSTM_gru_vlad256_256_0.80.pt'))
# audio_lstm_model = torch.load(os.path.join(prefix, 'Model/Classification/Audio3/BiLSTM_gru_vlad256_256_0.89.pt'))
# audio_lstm_model = torch.load(os.path.join(prefix, 'Model/Classification/Audio2/BiLSTM_gru_vlad256_256_0.65.pt'))

# model = BiLSTM(config['rnn_layers'], config['dropout'], config['num_classes'], \
#          config['hidden_dims'], config['embedding_size'])
         
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
# model_state_dict = audio_lstm_model.state_dict()
# model.load_state_dict(model_state_dict, strict=False)

def evaluate(model, test_idxs):
    model.eval()
    batch_idx = 1
    total_loss = 0
    pred = torch.empty(config['batch_size'], 1).type(torch.LongTensor)
    # X_test = audio_features[test_dep_idxs+test_non_idxs]
    # Y_test = audio_targets[test_dep_idxs+test_non_idxs]
    X_test = audio_features[test_idxs]
    Y_test = audio_targets[test_idxs]
    global max_train_acc, max_acc,max_f1
    for i in range(0, X_test.shape[0], config['batch_size']):
        if i + config['batch_size'] > X_test.shape[0]:
            x, y = X_test[i:], Y_test[i:]
        else:
            x, y = X_test[i:(i+config['batch_size'])], Y_test[i:(i+config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        else:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), Variable(torch.from_numpy(y))
        with torch.no_grad():
            output = model(x.squeeze(2))
        pred = torch.cat((pred, output.data.max(1, keepdim=True)[1]))
        
    y_test_pred, conf_matrix = model_performance(Y_test, pred[config['batch_size']:])
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}\n".format(f1_score))
    print('='*89)
    return precision, recall, f1_score


# evaluate(audio_features_test, fuse_targets_test, audio_lstm_model)
# evaluate(model)

idxs_paths = ['train_idxs_0.63_1.npy', 'train_idxs_0.65_2.npy', 'train_idxs_0.60_3.npy']
audio_model_paths = ['BiLSTM_gru_vlad256_256_0.67_1.pt', 'BiLSTM_gru_vlad256_256_0.67_2.pt', 'BiLSTM_gru_vlad256_256_0.63_3.pt']
ps, rs, fs = [], [], []
for fold in range(3):
    train_idxs_tmp = np.load(os.path.join(prefix, 'Features/TextWhole/{}'.format(idxs_paths[fold])), allow_pickle=True)
    test_idxs_tmp = list(set(list(audio_dep_idxs)+list(audio_non_idxs)) - set(train_idxs_tmp))
    audio_lstm_model = torch.load(os.path.join(prefix, 'Model/ClassificationWhole/Audio/{}'.format(audio_model_paths[fold])))

    train_idxs, test_idxs = [], []
    for idx in train_idxs_tmp:
        if idx in audio_dep_idxs:
            feat = audio_features[idx]
            count = 0
            resample_idxs = [0,1,2,3,4,5]
            for i in itertools.permutations(feat, feat.shape[0]):
                if count in resample_idxs:
                    audio_features = np.vstack((audio_features, np.expand_dims(list(i), 0)))
                    audio_targets = np.hstack((audio_targets, 1))
                    train_idxs.append(len(audio_features)-1)
                count += 1
        else:
            train_idxs.append(idx)

    for idx in test_idxs_tmp:
        if idx in audio_dep_idxs:
            feat = audio_features[idx]
            count = 0
            # resample_idxs = random.sample(range(6), 4)
            resample_idxs = [0,1,4,5]
            for i in itertools.permutations(feat, feat.shape[0]):
                if count in resample_idxs:
                    audio_features = np.vstack((audio_features, np.expand_dims(list(i), 0)))
                    audio_targets = np.hstack((audio_targets, 1))
                    test_idxs.append(len(audio_features)-1)
                count += 1
        else:
            test_idxs.append(idx)
    p, r, f = evaluate(audio_lstm_model, test_idxs)
    ps.append(p)
    rs.append(r)
    fs.append(f)
print('precison: {} \n recall: {} \n f1 score: {}'.format(np.mean(ps), np.mean(rs), np.mean(fs)))


