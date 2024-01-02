# This repository is the official implementation of [Uncertainty-Aware Label Contrastive Distribution Learning for Automatic Depression Detection](https://ieeexplore.ieee.org/abstract/document/10251764) 
## Abstract
Depression is one of the most common mental illnesses, affecting people’s quality of life and posing a risk to their health. Low-cost and objective automatic depression detection (ADD) is becoming increasingly critical. However, existing ADD methods usually treat depression detection as a regression problem for predicting patient health questionnaire-8 (PHQ-8) scores, ignoring the scores’ ambiguity caused by multiple questionnaire issues. To effectively leverage the score labels, we propose an uncertainty-aware label contrastive and distribution learning (ULCDL) method to estimate PHQ-8 scores, thus detecting depression automatically. ULCDL first simulates the ambiguity within PHQ-8 scores by converting single-valued scores into discrete label distributions. Afterward, it learns to predict the PHQ-8 score distribution by minimizing the Kullback–Leibler (KL) divergence between the score distribution and the discrete label distribution. Finally, the predicted PHQ-8 score distribution outperforms the PHQ-8 score in ADD. Moreover, label-based contrastive learning (LBCL) is introduced to facilitate the model for learning common features related to depression in multimodal data. A multibranch fusion module is proposed to align and fuse multimodal data for better exploring the uncertainty of PHQ-8 labels. The proposed method is evaluated on the publicly available DAIC-WOZ dataset. Experiment results show that ULCDL outperforms regression-based depression detection methods and achieves state-of-the-art performance.

## Requirements
CUDA 11.3

Python = 3.8+

Pytorch = 1.19+

## Data
This thesis uses the [DAIC-WOZ depression database](https://dcapswoz.ict.usc.edu/). In it you can send a request and receive your own username and password to access the database.

Preprocessing of data using database_generation.py

## Training
python main.py

