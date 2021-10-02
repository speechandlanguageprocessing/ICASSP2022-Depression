from fuse_net_whole import fusion_net, config, model_performance
import os
import numpy as np
import torch
from torch.autograd import Variable
import itertools

prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))
idxs_paths = ['train_idxs_0.63_1.npy', 'train_idxs_0.65_2.npy', 'train_idxs_0.60_3.npy']
text_model_paths = ['BiLSTM_128_0.67_1.pt', 'BiLSTM_128_0.66_2.pt', 'BiLSTM_128_0.66_3.pt']
audio_model_paths = ['BiLSTM_gru_vlad256_256_0.63_1.pt', 'BiLSTM_gru_vlad256_256_0.65_2.pt', 'BiLSTM_gru_vlad256_256_0.60_3.pt']
fuse_model_paths = ['fuse_0.69_1.pt', 'fuse_0.68_2.pt', 'fuse_0.62_3.pt']
text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_clf_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_clf_avg.npz'))['arr_0']
audio_features = np.squeeze(np.load(os.path.join(prefix, 'Features/AudioWhole/whole_samples_clf_256.npz'))['arr_0'], axis=2)
audio_targets = np.load(os.path.join(prefix, 'Features/AudioWhole/whole_labels_clf_256.npz'))['arr_0']
fuse_features = [[audio_features[i], text_features[i]] for i in range(text_features.shape[0])]
fuse_targets = text_targets
fuse_dep_idxs = np.where(text_targets == 1)[0]
fuse_non_idxs = np.where(text_targets == 0)[0]

def evaluate(model, test_idxs):
    model.eval()
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
        pred = torch.cat((pred, output.data.max(1, keepdim=True)[1]))
        
    y_test_pred, conf_matrix = model_performance(Y_test, pred[config['batch_size']:])
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

    return precision, recall, f1_score

ps, rs, fs = [], [], []
for fold in range(3):
    train_idxs_tmp = np.load(os.path.join(prefix, 'Features/TextWhole/{}'.format(idxs_paths[fold])), allow_pickle=True)
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
    
    fuse_model = torch.load(os.path.join(prefix, 'Model/ClassificationWhole/Fuse/{}'.format(fuse_model_paths[fold])))
    p, r, f = evaluate(fuse_model, test_idxs)
    ps.append(p)
    rs.append(r)
    fs.append(f)
print('precison: {} \n recall: {} \n f1 score: {}'.format(np.mean(ps), np.mean(rs), np.mean(fs)))
