# ICASSP2022-Depression
Automatic Depression Detection: a GRU/ BiLSTM-based Model and An Emotional Audio-Textual Corpus

https://arxiv.org/pdf/2202.08210.pdf

https://ieeexplore.ieee.org/abstract/document/9746569/

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


## Dataset: EATD-Corpus

The EATD-Corpus is a dataset consist of audio and text files of 162 volunteers who received counseling.

### How to download
The EATD-Corpus can be downloaded at https://1drv.ms/u/s!AsGVGqImbOwYhHUHcodFC3xmKZKK?e=mCT5oN. Password: Ymj26Uv5

### How to use

Training set contains data from 83 volunteers (19 depressed and 64 non-depressed).

Validation set contains data from 79 volunteers (11 depressed and 68 non-depressed).

Each folder contains depression data for one volunteer.

- {positive/negative/neutral}.wav: Raw audio in wav
- {positive/negative/neutral}_out.wav: Preprocessed audio. Preprocessing operations include denoising and de-muting
- {positive/negative/neutral}.txt: Audio translation
- label.txt: Raw SDS score
- new_label.txt: Standard SDS score (Raw SDS score multiplied by 1.25)
