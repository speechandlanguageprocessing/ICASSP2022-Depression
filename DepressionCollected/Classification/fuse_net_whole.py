
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
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

text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_clf_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_clf_avg.npz'))['arr_0']
audio_features = np.squeeze(np.load(os.path.join(prefix, 'Features/AudioWhole/whole_samples_clf_256.npz'))['arr_0'], axis=2)
audio_targets = np.load(os.path.join(prefix, 'Features/AudioWhole/whole_labels_clf_256.npz'))['arr_0']
fuse_features = [[audio_features[i], text_features[i]] for i in range(text_features.shape[0])]
fuse_targets = text_targets

fuse_dep_idxs = np.where(text_targets == 1)[0]
fuse_non_idxs = np.where(text_targets == 0)[0]

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
    
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
        # self.init_weight()

    def init_weight(net):
        for name, param in net.named_parameters():
            if not 'ln' in name:
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

        # self.lstm_net_audio = nn.LSTM(self.embedding_size,
        #                         self.hidden_dims,
        #                         num_layers=self.rnn_layers,
        #                         dropout=self.dropout,
        #                         bidirectional=self.bidirectional,
        #                         batch_first=True)
        self.lstm_net_audio = nn.GRU(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout, batch_first=True)

        self.ln = nn.LayerNorm(self.embedding_size)

        # FC层
        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1)
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
        x = self.ln(x)
        x, _ = self.lstm_net_audio(x)
        x = x.mean(dim=1)
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

        self.ln = nn.LayerNorm(self.audio_embed_size)
        
        # ============================= AudioBiLSTM =============================

        # ============================= last fc layer =============================
        # self.bn = nn.BatchNorm1d(self.text_hidden_dims + self.audio_hidden_dims)
        # modal attention
        self.modal_attn = nn.Linear(self.text_hidden_dims + self.audio_hidden_dims, self.text_hidden_dims + self.audio_hidden_dims, bias=False)
        self.fc_final = nn.Sequential(
            nn.Linear(self.text_hidden_dims + self.audio_hidden_dims, self.num_classes, bias=False),
            # nn.ReLU(),
            nn.Softmax(dim=1),
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
            x_audio = self.ln(x_audio)
            x_audio, _ = self.lstm_net_audio(x_audio)
            x_audio = x_audio.sum(dim=1)
            audio_feature = self.fc_audio(x_audio)

        # ============================= AudioBiLSTM =============================
        return (text_feature, audio_feature)
        
    def forward(self, x): 
        # x = self.bn(x)
        # modal_weights = torch.softmax(self.modal_attn(x), dim=1)
        # modal_weights = self.modal_attn(x)
        # x = (modal_weights * x)
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
        l = nn.CrossEntropyLoss()
        target = torch.tensor(target)
        # l = nn.BCEWithLogitsLoss()
        # target = F.one_hot(target, num_classes=2).type(torch.FloatTensor)
        # print('y: {}\npred_audio: {}\npred_text: {}\n'.format(target, pred_audio.data.max(1, keepdim=True)[1], pred_text.data.max(1, keepdim=True)[1]))
        # return l(pred_text, target) + l(pred_audio, target) + \
        #         config['lambda']*torch.norm(weight[:, :config['text_hidden_dims']]) + \
        #         config['lambda']*torch.norm(weight[:, config['text_hidden_dims']:])  
        # a = F.softmax(pred_text, dim=1) + F.softmax(pred_audio, dim=1)
        return l(pred_text, target) + l(pred_audio, target)
    

config = {
    'num_classes': 2,
    'dropout': 0.3,
    'rnn_layers': 2,
    'audio_embed_size': 256,
    'text_embed_size': 1024,
    'batch_size': 2,
    'epochs': 100,
    'learning_rate': 8e-6,
    'audio_hidden_dims': 256,
    'text_hidden_dims': 128,
    'cuda': False,
    'lambda': 1e-5,
}

model = fusion_net(config['text_embed_size'], config['text_hidden_dims'], config['rnn_layers'], \
    config['dropout'], config['num_classes'], config['audio_hidden_dims'], config['audio_embed_size'])

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()
criterion = MyLoss()

def train(epoch, train_idxs):
    global max_train_acc, train_acc
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    X_train = []
    Y_train = []
    for idx in train_idxs:
        X_train.append(fuse_features[idx])
        Y_train.append(fuse_targets[idx])
    for i in range(0, len(X_train), config['batch_size']):
        if i + config['batch_size'] > len(X_train):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+config['batch_size'])], Y_train[i:(i+config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        text_feature, audio_feature = model.pretrained_feature(x)
        # text_feature = torch.from_numpy(ss.fit_transform(text_feature.numpy()))
        # audio_feature = torch.from_numpy(ss.fit_transform(audio_feature.numpy()))
        # concat_x = torch.cat((audio_feature, text_feature), dim=1)
        concat_x = torch.cat((text_feature, audio_feature), dim=1)
        # dot_x = text_feature.mul(audio_feature)
        # add_x = text_feature.add(audio_feature)
        output = model(concat_x)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(torch.tensor(y).data.view_as(pred)).cpu().sum()
        # loss = criterion(output, torch.tensor(y))
        loss = criterion(text_feature, audio_feature, y, model)
        # 后向传播调整参数
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        batch_idx += 1
        # loss.item()能够得到张量中的元素值
        total_loss += loss.item()
    cur_loss = total_loss
    max_train_acc = correct
    train_acc = correct
    print('Train Epoch: {:2d}\t Learning rate: {:.4f}\tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)\n '.format(
                epoch, config['learning_rate'], cur_loss/len(X_train), correct, len(X_train),
        100. * correct / len(X_train)))


def evaluate(model, test_idxs, fold, train_idxs):
    model.eval()
    batch_idx = 1
    total_loss = 0
    pred = torch.empty(config['batch_size'], 1).type(torch.LongTensor)
    X_test = []
    Y_test = []
    for idx in test_idxs:
        X_test.append(fuse_features[idx])
        Y_test.append(fuse_targets[idx])
    global max_train_acc, max_acc,max_f1
    for i in range(0, len(X_test), config['batch_size']):
        if i + config['batch_size'] > len(X_test):
            x, y = X_test[i:], Y_test[i:]
        else:
            x, y = X_test[i:(i+config['batch_size'])], Y_test[i:(i+config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        text_feature, audio_feature = model.pretrained_feature(x)
        with torch.no_grad():
            # concat_x = torch.cat((audio_feature, text_feature), dim=1)
            audio_feature_norm = (audio_feature - audio_feature.mean())/audio_feature.std()
            text_feature_norm = (text_feature - text_feature.mean())/text_feature.std()
            concat_x = torch.cat((text_feature, audio_feature), dim=1)
            output = model(concat_x)
        # loss = criterion(output, torch.tensor(y))
        loss = criterion(text_feature, audio_feature, y, model)
        pred = torch.cat((pred, output.data.max(1, keepdim=True)[1]))
        total_loss += loss.item()
        
    y_test_pred, conf_matrix = model_performance(Y_test, pred[config['batch_size']:])
    
    print('\nTest set: Average loss: {:.4f}'.format(total_loss/len(X_test)))
    # custom evaluation metrics
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
    
    if max_f1 < f1_score and max_train_acc >= len(train_idxs)*0.9 and f1_score > 0.61:
        max_f1 = f1_score
        max_acc = accuracy
        save(model, os.path.join(prefix, 'Model/ClassificationWhole/Fuse/fuse_{:.2f}_{}'.format(max_f1, fold)))
        print('*'*64)
        print('model saved: f1: {}\tacc: {}'.format(max_f1, max_acc))
        print('*'*64)
    return total_loss

if __name__ == '__main__':
    idxs_paths = ['train_idxs_0.63_1.npy', 'train_idxs_0.65_2.npy', 'train_idxs_0.60_3.npy']
    text_model_paths = ['BiLSTM_128_0.64_1.pt', 'BiLSTM_128_0.66_2.pt', 'BiLSTM_128_0.62_3.pt']
    audio_model_paths = ['BiLSTM_gru_vlad256_256_0.67_1.pt', 'BiLSTM_gru_vlad256_256_0.67_2.pt', 'BiLSTM_gru_vlad256_256_0.63_3.pt']
    for fold in range(1, 4):
        # if fold != 2:
        #     continue
        train_idxs_tmp = np.load(os.path.join(prefix, 'Features/TextWhole/{}'.format(idxs_paths[fold-1])), allow_pickle=True)
        test_idxs_tmp = list(set(list(fuse_dep_idxs)+list(fuse_non_idxs)) - set(train_idxs_tmp))
        resample_idxs = list(range(6))

        train_idxs, test_idxs = [], []
        # depression data augmentation
        for idx in train_idxs_tmp:
            if idx in fuse_dep_idxs:
                feat = fuse_features[idx]
                audio_perm = itertools.permutations(feat[0], 3)
                text_perm = itertools.permutations(feat[1], 3)
                count = 0
                for fuse_perm in zip(audio_perm, text_perm):
                    if count in resample_idxs:
                        fuse_features.append(fuse_perm)
                        fuse_targets = np.hstack((fuse_targets, 1))
                        train_idxs.append(len(fuse_features)-1)
                    count += 1
            else:
                train_idxs.append(idx)

        for idx in test_idxs_tmp:
            if idx in fuse_dep_idxs:
                feat = fuse_features[idx]
                audio_perm = itertools.permutations(feat[0], 3)
                text_perm = itertools.permutations(feat[1], 3)
                count = 0
                resample_idxs = [0,1,4,5]
                for fuse_perm in zip(audio_perm, text_perm):
                    if count in resample_idxs:
                        fuse_features.append(fuse_perm)
                        fuse_targets = np.hstack((fuse_targets, 1))
                        test_idxs.append(len(fuse_features)-1)
                    count += 1
            else:
                test_idxs.append(idx)

        text_lstm_model = torch.load(os.path.join(prefix, 'Model/ClassificationWhole/Text/{}'.format(text_model_paths[fold-1])))
        audio_lstm_model = torch.load(os.path.join(prefix, 'Model/ClassificationWhole/Audio/{}'.format(audio_model_paths[fold-1])))
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

        model_state_dict['ln.weight'] = audio_lstm_model.state_dict()['ln.weight']
        model_state_dict['ln.bias'] = audio_lstm_model.state_dict()['ln.bias']
        model.load_state_dict(text_lstm_model.state_dict(), strict=False)
        # model.load_state_dict(audio_lstm_model.state_dict(), strict=False)
        model.load_state_dict(model_state_dict, strict=False)
            
        for param in model.parameters():
            param.requires_grad = False

        model.fc_final[0].weight.requires_grad = True
        # model.fc_final[0].bias.requires_grad = True
        # model.modal_attn.weight.requires_grad = True

        max_f1 = -1
        max_acc = -1
        max_train_acc = -1

        for ep in range(1, config['epochs']):
            train(ep, train_idxs)
            tloss = evaluate(model, test_idxs, fold, train_idxs)