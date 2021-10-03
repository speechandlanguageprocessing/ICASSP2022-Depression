# ICASSP2022-Depression
Automatic Depression Detection: a GRU/ BiLSTM-based Model and An Emotional Audio-Textual Corpus


## Code

- Regression
  - audio_bilstm_perm.py: train audio network 
  - text_bilstm_perm.py: train text network 
  - fuse_net.py: train multi-modal network
- Classification
  - audio_features_whole.py: extract audio features
  - text_features_whole.py: extract text features
  - audio_gru_whole.py: train audio network 
  - text_bilstm_whole.py: train text network
  - fuse_net_whole.py: train fuse network


## Dataset

- EATD-Corpus: a dataset consis of audio and text files of 162 volunteers who received counseling

### How to use EATD-Corpus

There are 162 folders in the dataset. Each folder contains depression data for one volunteer.

- {positive/negative/neutral}.wav: Raw audio in wav
- {positive/negative/neutral}.mp3: Raw audio in mp3
- {positive/negative/neutral}_out.wav: Preprocessed audio. Preprocessing operations include denoising and de-muting
- {positive/negative/neutral}.txt: Audio translation
- {positive/negative/neutral}_seg.txt: Audio translation with word segmentation
- {positive/negative/neutral}_out.csv: Audio features extracted with COVAREP toolkits
- label.txt: Raw SDS score
- new_label.txt: SDS score multiplying 1.25

### How to download EATD-Corpus
 https://pan.baidu.com/s/1dIZBRZxLaFjLEsis47b7ag with extraction code 2022
