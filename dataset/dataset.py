from itertools import count
import os
import numpy as np
import pandas as pd
from skimage import transform

import torch
from torch.utils.data import Dataset


class DepressionDataset(Dataset):


    def __init__(self,
                 root_dir,
                 mode,
                 use_mel_spectrogram=True,
                 visual_with_gaze=True,
                 transform=None):
        super(DepressionDataset, self).__init__()
        self.mode = mode
        self.root_dir = root_dir
        self.use_mel_spectrogram = use_mel_spectrogram
        self.visual_with_gaze = visual_with_gaze
        self.transform = transform

        if mode == 'train':
            # store ground truth
            self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'))
            self.gender_gt = np.load(os.path.join(self.root_dir, 'gender_gt.npy'))
            self.binary_gt = np.load(os.path.join(self.root_dir, 'binary_gt.npy'))
            self.score_gt = np.load(os.path.join(self.root_dir, 'score_gt.npy'))
            self.subscores_gt = np.load(os.path.join(self.root_dir, 'subscores_gt.npy'))

        elif mode == 'validation':

            # store ground truth
            self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'))
            self.gender_gt = np.load(os.path.join(self.root_dir, 'gender_gt.npy'))
            self.binary_gt = np.load(os.path.join(self.root_dir, 'binary_gt.npy'))
            self.score_gt = np.load(os.path.join(self.root_dir, 'score_gt.npy'))
            self.subscores_gt = np.load(os.path.join(self.root_dir, 'subscores_gt.npy'))

        elif mode == 'test':

            # store ground truth
            self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'))
            self.gender_gt = np.load(os.path.join(self.root_dir, 'gender_gt.npy'))
            self.binary_gt = np.load(os.path.join(self.root_dir, 'binary_gt.npy'))
            self.score_gt = np.load(os.path.join(self.root_dir, 'score_gt.npy'))

    def __len__(self):
        return len(self.IDs)

    def __iter__(self):
        return iter(self.IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get visual feature
        if self.visual_with_gaze:
            fkps_path = os.path.join(self.root_dir, 'facial_keypoints', 'only_coordinate')
            gaze_path = os.path.join(self.root_dir, 'gaze_vectors', 'only_coordinate')

            # load and create final visual feature
            fkps_file = np.sort(os.listdir(fkps_path))[idx]
            gaze_file = np.sort(os.listdir(gaze_path))[idx]
            fkps = np.load(os.path.join(fkps_path, fkps_file))
            gaze = np.load(os.path.join(gaze_path, gaze_file))
            visual = np.concatenate((fkps, gaze), axis=1)
        else:
            fkps_path = os.path.join(self.root_dir, 'facial_keypoints', 'only_coordinate')
            # load and create final visual feature
            fkps_file = np.sort(os.listdir(fkps_path))[idx]
            visual = np.load(os.path.join(fkps_path, fkps_file))


        # get audio feature
        if self.use_mel_spectrogram:
            audio_path = os.path.join(self.root_dir, 'audio', 'mel-spectrogram')
        else:
            audio_path = os.path.join(self.root_dir, 'audio', 'spectrogram')
        audio_file = np.sort(os.listdir(audio_path))[idx]
        audio = np.load(os.path.join(audio_path, audio_file))  # shape: frequency_bins x num_sample (80 x 1800)

        # get text feature
        text_path = os.path.join(self.root_dir, 'text', 'sentence_embeddings')
        text_file = np.sort(os.listdir(text_path))[idx]
        text = np.load(os.path.join(text_path, text_file))

        # summary
        if self.mode == 'test':
            session = {'ID': self.IDs[idx],
                       'gender_gt': self.gender_gt[idx],
                       'binary_gt': self.binary_gt[idx],
                       'score_gt': self.score_gt[idx],
                       'visual': visual,
                       'audio': audio,
                       'text': text}
        else:
            session = {'ID': self.IDs[idx],
                       'gender_gt': self.gender_gt[idx],
                       'binary_gt': self.binary_gt[idx],
                       'score_gt': self.score_gt[idx],
                       'subscores_gt': self.subscores_gt[idx],
                       'visual': visual,
                       'audio': audio,
                       'text': text}

        if self.transform:
            session = self.transform(session)

        return session


class Padding(object):
    ''' pad zero to each feature matrix so that they all have the same size '''

    def __init__(self, audio_output_size=(80, 2000)):
        super(Padding, self).__init__()

        assert isinstance(audio_output_size, (int, tuple))
        self.audio_output_size = audio_output_size

    def __call__(self, session):
        padded_session = session
        audio = session['audio']
        visual = session['visual']

        # audio padding along width dimension
        if isinstance(self.audio_output_size, int):
            h, w = audio.shape
            h_v, w_v, c_v = visual.shape
            new_w = self.audio_output_size if w > self.audio_output_size else w
            padded_audio = np.zeros((h, self.audio_output_size))
            padded_visual = np.zeros((self.audio_output_size, w_v, c_v))
            padded_audio[:h, :new_w] = audio[:h, :new_w]
            padded_visual[:new_w, :w_v, :c_v] = visual[:new_w, :w_v, :c_v]

        # audio padding along both heigh and width dimension
        else:
            h, w = audio.shape
            new_h = self.audio_output_size[0] if h > self.audio_output_size[0] else h
            new_w = self.audio_output_size[1] if w > self.audio_output_size[1] else w
            padded_audio = np.zeros(self.audio_output_size)
            padded_audio[:new_h, :new_w] = audio[:new_h, :new_w]

        # summary
        padded_session['audio'] = padded_audio
        padded_session['visual'] = padded_visual

        return padded_session


class Rescale(object):


    def __init__(self, output_size=(80, 900)):
        assert isinstance(output_size, (int, tuple, list))

        if type(output_size) == list:
            assert len(output_size) == 2, "Rescale output size should be 2 dimensional"

        self.output_size = output_size

    def __call__(self, session):
        rescaled_session = session
        audio = session['audio']

        h, w = audio.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        rescaled_audio = transform.resize(audio, (new_h, new_w))

        # summary
        rescaled_session['audio'] = rescaled_audio

        return rescaled_session


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Arguments:
        output_size:(tuple or int), Desired output size. 
        If int, square crop is made.
    """

    def __init__(self, output_size=(224, 224)):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, session):
        cropped_session = session
        audio = session['audio']

        h, w = audio.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_audio = audio[top:top + new_h, left:left + new_w]

        # summary
        cropped_session['audio'] = cropped_audio

        return cropped_session


class ToTensor(object):
    """Convert ndarrays in sample to Tensors or np.int to torch.tensor."""

    def __init__(self, mode):
        # assert mode in ["train", "validation", "test"], \
        #     "Argument --mode could only be ['train', 'validation', 'test']"

        self.mode = mode

    def __call__(self, session):
        if self.mode == 'test':
            converted_session = {'ID': session['ID'],
                                 'gender_gt': torch.tensor(session['gender_gt']).type(torch.FloatTensor),
                                 'binary_gt': torch.tensor(session['binary_gt']).type(torch.LongTensor),
                                 'score_gt': torch.tensor(session['score_gt']).type(torch.FloatTensor),
                                 'visual': torch.from_numpy(session['visual']).type(torch.FloatTensor),
                                 'audio': torch.from_numpy(session['audio']).type(torch.FloatTensor),
                                 'text': torch.from_numpy(session['text']).type(torch.FloatTensor)}

        else:
            converted_session = {'ID': session['ID'],
                                 'gender_gt': torch.tensor(session['gender_gt']).type(torch.FloatTensor),
                                 'binary_gt': torch.tensor(session['binary_gt']).type(torch.LongTensor),
                                 'score_gt': torch.tensor(session['score_gt']).type(torch.FloatTensor),
                                 'subscores_gt': torch.from_numpy(session['subscores_gt']).type(torch.FloatTensor),
                                 'visual': torch.from_numpy(session['visual']).type(torch.FloatTensor),
                                 'audio': torch.from_numpy(session['audio']).type(torch.FloatTensor),
                                 'text': torch.from_numpy(session['text']).type(torch.FloatTensor)}

        return converted_session
