import os
import sys
sys.path.append('/Users/linlin/Desktop/DepressionCollected')
from Classification.audio_features_whole import wav2vlad

import numpy as np
import pandas as pd
import wave

prefix = os.getcwd()
train_split_df = pd.read_csv(os.path.join(prefix, 'DAIC/train_split_Depression_AVEC2017.csv'))
test_split_df = pd.read_csv(os.path.join(prefix, 'DAIC/dev_split_Depression_AVEC2017.csv'))
train_split_num = train_split_df[['Participant_ID']]['Participant_ID'].tolist()
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()
train_split_clabel = train_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
test_split_clabel = test_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
train_split_rlabel = train_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()
test_split_rlabel = test_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()

with open('./queries.txt') as f:
    queries = f.readlines()

def identify_topics(sentence):
    for query in queries:
        query = query.strip('\n')
        sentence = sentence.strip('\n')
        if query == sentence:
            return True
    return False

def extract_features(number):
    transcript = pd.read_csv(os.path.join(prefix, 'DAIC/{0}_P/{0}_TRANSCRIPT.csv'.format(number)), sep='\t').fillna('')
    
    wavefile = wave.open(os.path.join(prefix, 'DAIC/{0}_P/{0}_AUDIO.wav'.format(number, 'r')))
    sr = wavefile.getframerate()
    nframes = wavefile.getnframes()
    wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
    
    response = ''
    start_time = 0
    stop_time = 0
    feats = []
    signal = []

    for t in transcript.itertuples():
        # 问题开始
        if getattr(t,'speaker') == 'Ellie' and (identify_topics(getattr(t,'value')) or 'i think i have asked everything' in getattr(t,'value')):
            # 初始化
            response = ''
            if len(signal) == 0:
                continue
            feats.append(wav2vlad(signal, sr))
            signal = []
        elif getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            start_time = int(getattr(t,'start_time')*sr)
            stop_time = int(getattr(t,'stop_time')*sr)
            response += (' ' + getattr(t,'value'))
            signal = np.hstack((signal, wave_data[start_time:stop_time].astype(np.float)))
    
    print(np.shape(feats))
    print('{}_P feature done'.format(number))
    return feats
    
# training set
audio_features_train = []
audio_ctargets_train = []
audio_rtargets_train = []

# test set
audio_features_test = []
audio_ctargets_test = []
audio_rtargets_test = []

# training set
for index in range(len(train_split_num)):
    feat = extract_features(train_split_num[index])
    audio_features_train.append(feat)
    audio_ctargets_train.append(train_split_clabel[index])
    audio_rtargets_train.append(train_split_rlabel[index])
    
print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features/train_samples_clf.npz'), audio_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features/train_samples_reg.npz'), audio_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features/train_labels_clf.npz'), audio_ctargets_train)
np.savez(os.path.join(prefix, 'DAICCode/Features/train_labels_reg.npz'), audio_rtargets_train)

# test set
for index in range(len(test_split_num)):
    feat = extract_features(test_split_num[index])
    audio_features_test.append(feat)
    audio_ctargets_test.append(test_split_clabel[index])
    audio_rtargets_test.append(test_split_rlabel[index])

print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features/test_samples_clf.npz'), audio_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features/test_samples_reg.npz'), audio_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features/test_labels_clf.npz'), audio_ctargets_test)
np.savez(os.path.join(prefix, 'DAICCode/Features/test_labels_reg.npz'), audio_rtargets_test)
