import os
import numpy as np
import pandas as pd
import wave
import librosa
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')


def create_folders(root_dir):
    folders = ['original_data', 'clipped_data', 'key_data']
    subfolders = ['no_gender_balance', 'gender_balance']
    subsubfolders = {'facial_keypoints': ['only_coordinate', 'coordinate+confidence'],
                     'gaze_vectors': ['only_coordinate', 'coordinate+confidence'],
                     'audio': ['spectrogram', 'mel-spectrogram'],
                     'text': ['sentence_embeddings']}

    os.makedirs(root_dir, exist_ok=True)
    for i in folders:
        for j in subfolders:
            for k, v in subsubfolders.items():
                for m in v:
                    # print(os.path.join(root, i, j,  k, m))
                    os.makedirs(os.path.join(root_dir, i, j, k, m), exist_ok=True)


def min_max_scaler(data):
    '''recale the data, which is a 2D matrix, to 0-1'''
    return (data - data.min()) / (data.max() - data.min())


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def pre_check(data_df):
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    data_np = data_df.to_numpy()
    data_min = data_np[np.where(~(np.isnan(data_np[:, 4:])))].min()
    data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
    return data_df


def load_gaze(gaze_path):
    gaze_df = pre_check(pd.read_csv(gaze_path, low_memory=False))
    # process into format TxVxC
    gaze_conf = gaze_df[' confidence'].to_numpy()
    gaze_coor = gaze_df.iloc[:, 4:].to_numpy().reshape(len(gaze_df), 4, 3)  # 4 gaze vectors, 3 axes
    T, V, C = gaze_coor.shape

    # initialize the final gaze_3D which contains coordinate and confidence score
    gaze_final = np.zeros((T, V, C + 1))

    gaze_final[:, :, :3] = gaze_coor
    for i in range(V):
        gaze_final[:, i, 3] = gaze_conf

    return gaze_coor, gaze_final


def load_keypoints(keypoints_path):
    fkps_df = pre_check(pd.read_csv(keypoints_path, low_memory=False))
    # process into format TxVxC
    fkps_conf = fkps_df[' confidence'].to_numpy()
    x_coor = min_max_scaler(fkps_df[fkps_df.columns[4: 72]].to_numpy())
    y_coor = min_max_scaler(fkps_df[fkps_df.columns[72: 140]].to_numpy())
    z_coor = min_max_scaler(fkps_df[fkps_df.columns[140: 208]].to_numpy())
    fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)
    T, V, C = fkps_coor.shape

    # initialize the final facial key points which contains coordinate and confidence score
    fkps_final = np.zeros((T, V, C + 1))

    fkps_final[:, :, :3] = fkps_coor
    for i in range(V):
        fkps_final[:, i, 3] = fkps_conf

    return fkps_coor, fkps_final


def load_audio(audio_path):
    wavefile = wave.open(audio_path)
    audio_sr = wavefile.getframerate()
    n_samples = wavefile.getnframes()
    signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)

    return signal.astype(float), audio_sr


def visual_clipping(visual_data, visual_sr, text_df):
    counter = 0
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t, 'value'):
                continue
            else:
                start = getattr(t, 'start_time')
                stop = getattr(t, 'stop_time')
                start_sample = int(start * visual_sr)
                stop_sample = int(stop * visual_sr)
                if counter == 0:
                    edited_vdata = visual_data[start_sample:stop_sample]
                else:
                    edited_vdata = np.vstack((edited_vdata, visual_data[start_sample:stop_sample]))

                counter += 1

    return edited_vdata


def visual_key(visual_data, visual_sr, text_df):
    counter = 0
    # keywords = [['advice', 'yourself'], ['anything', 'regret'], ['argued', 'someone'], ['controlling', 'temper'],
    #             ['diagnosed', 'depression'], ['diagnosed', 'ptsd'], ['doing', 'today'], ['dream', 'job'],
    #             ['easy', 'sleep'], ['feeling', 'lately'], ['friend', 'describe'], ['last', 'happy'],
    #             ['proud', 'life'], ['study', 'school'], ['travel', 'lot']]
    keywords = [['advice', 'yourself'], ['anything', 'regret'], ['argued', 'someone'], ['controlling', 'temper'],
                ['dream', 'job'], ['easy', 'sleep'], ['feeling', 'lately'], ['friend', 'describe'], ['last', 'happy'],
                ['proud', 'life']]
    key_ = False
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t, 'value'):
                key_ = False
                continue
            else:
                if key_:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * visual_sr)
                    stop_sample = int(stop * visual_sr)
                    if counter == 0:
                        edited_vdata = visual_data[start_sample:stop_sample]
                    else:
                        edited_vdata = np.vstack((edited_vdata, visual_data[start_sample:stop_sample]))
                    key_ = False
                    counter += 1
                else:
                    continue
        else:
            for key in keywords:
                if key[0] in getattr(t, 'value') and key[1] in getattr(t, 'value'):
                    key_ = True

    return edited_vdata


def audio_clipping(audio, audio_sr, text_df, zero_padding=False):
    if zero_padding:
        edited_audio = np.zeros(audio.shape[0])
        for t in text_df.itertuples():
            if getattr(t, 'speaker') == 'Participant':
                if 'scrubbed_entry' in getattr(t, 'value'):
                    continue
                else:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * audio_sr)
                    stop_sample = int(stop * audio_sr)
                    edited_audio[start_sample:stop_sample] = audio[start_sample:stop_sample]

        # cut head and tail of interview
        first_start = text_df['start_time'][0]
        last_stop = text_df['stop_time'][len(text_df) - 1]
        edited_audio = edited_audio[int(first_start * audio_sr):int(last_stop * audio_sr)]

    else:
        edited_audio = []
        for t in text_df.itertuples():
            if getattr(t, 'speaker') == 'Participant':
                if 'scrubbed_entry' in getattr(t, 'value'):
                    continue
                else:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * audio_sr)
                    stop_sample = int(stop * audio_sr)
                    edited_audio = np.hstack((edited_audio, audio[start_sample:stop_sample]))

    return edited_audio


def audio_key(audio, audio_sr, text_df, zero_padding=False):
    edited_audio = []
    # keywords = [['advice', 'yourself'], ['anything', 'regret'], ['argued', 'someone'], ['controlling', 'temper'],
    #             ['diagnosed', 'depression'], ['diagnosed', 'ptsd'], ['doing', 'today'], ['dream', 'job'],
    #             ['easy', 'sleep'], ['feeling', 'lately'], ['friend', 'describe'], ['last', 'happy'],
    #             ['proud', 'life'], ['study', 'school'], ['travel', 'lot']]
    keywords = [['advice', 'yourself'], ['anything', 'regret'], ['argued', 'someone'], ['controlling', 'temper'],
                ['dream', 'job'], ['easy', 'sleep'], ['feeling', 'lately'], ['friend', 'describe'], ['last', 'happy'],
                ['proud', 'life']]
    key_ = False
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t, 'value'):
                key_ = False
                continue
            else:
                if key_:
                    start = getattr(t, 'start_time')
                    stop = getattr(t, 'stop_time')
                    start_sample = int(start * audio_sr)
                    stop_sample = int(stop * audio_sr)
                    edited_audio = np.hstack((edited_audio, audio[start_sample:stop_sample]))
                    key_ = False
                else:
                    continue
        else:
            for key in keywords:
                if key[0] in getattr(t, 'value') and key[1] in getattr(t, 'value'):
                    key_ = True

    return edited_audio


def convert_spectrogram(audio, frame_size=2048, hop_size=533):
    # extracting with Short-Time Fourier Transform
    S_scale = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)
    spectrogram = np.abs(S_scale) ** 2
    # convert amplitude to DBs
    log_spectrogram = librosa.power_to_db(spectrogram)

    return log_spectrogram  # in dB


def convert_mel_spectrogram(audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80):
    mel_spectrogram = librosa.feature.melspectrogram(audio,
                                                     sr=audio_sr,
                                                     n_fft=frame_size,
                                                     hop_length=hop_size,
                                                     n_mels=num_mel_bands)
    # convert amplitude to DBs
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    return log_mel_spectrogram  # in dB


def sentence_embedding(text_df, model):
    sentences = []
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t, 'value'):
                continue
            else:
                sentences.append(getattr(t, 'value'))

    return model.encode(sentences)


def sentence_embedding_key(text_df, model):
    sentences = []
    # keywords = [['advice', 'yourself'], ['anything', 'regret'], ['argued', 'someone'], ['controlling', 'temper'],
    #             ['diagnosed', 'depression'], ['diagnosed', 'ptsd'], ['doing', 'today'], ['dream', 'job'],
    #             ['easy', 'sleep'], ['feeling', 'lately'], ['friend', 'describe'], ['last', 'happy'],
    #             ['proud', 'life'], ['study', 'school'], ['travel', 'lot']]
    keywords = [['advice', 'yourself'], ['anything', 'regret'], ['argued', 'someone'], ['controlling', 'temper'],
                ['dream', 'job'], ['easy', 'sleep'], ['feeling', 'lately'], ['friend', 'describe'], ['last', 'happy'],
                ['proud', 'life']]
    key_ = False
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t, 'value'):
                key_ = False
                continue
            else:
                if key_:
                    sentences.append(getattr(t, 'value'))
                    key_ = False
                else:
                    continue
        else:
            for key in keywords:
                if key[0] in getattr(t, 'value') and key[1] in getattr(t, 'value'):
                    key_ = True

    return model.encode(sentences)


def get_num_frame(data, frame_size, hop_size):
    T = data.shape[0]
    if (T - frame_size) % hop_size == 0:
        num_frame = (T - frame_size) // hop_size + 1
    else:
        num_frame = (T - frame_size) // hop_size + 2
    return num_frame


def get_text_hop_size(text, frame_size, num_frame):
    T = text.shape[0]
    return (T - frame_size) // (num_frame - 1)


def visual_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple()
        size = size + (pad_size,) + data.shape[1:]
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def visual_padding_key(data, pad_size):
    if data.shape[0] < pad_size:
        size = tuple()
        size = size + (pad_size,) + data.shape[1:]
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def audio_padding(data, pad_size):
    if data.shape[1] != pad_size:
        size = tuple((data.shape[0], pad_size))
        padded_data = np.zeros(size)
        padded_data[:, :data.shape[1]] = data
    else:
        padded_data = data

    return padded_data


def audio_padding_key(data, pad_size):
    if data.shape[0] < pad_size:
        size = tuple((data.shape[0], pad_size))
        padded_data = np.zeros(size)
        padded_data[:, :data.shape[1]] = data
    else:
        padded_data = data

    return padded_data


def text_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple((pad_size, data.shape[1]))
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def text_padding_key(data, pad_size):
    if data.shape[0] < pad_size:
        size = tuple((pad_size, data.shape[1]))
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def random_shift_fkps(fkps_coor, fkps_coor_conf):
    shifted_fc = np.copy(fkps_coor)
    shifted_fcc = np.copy(fkps_coor_conf)

    for i in range(3):
        factor = np.random.uniform(-0.05, 0.05)
        shifted_fc[:, :, i] = shifted_fc[:, :, i] + factor
        shifted_fcc[:, :, i] = shifted_fcc[:, :, i] + factor

    return shifted_fc, shifted_fcc


def sliding_window(fkps_coor, fkps_coor_conf, gaze_coor, gaze_coor_conf,
                   spectro, mel_spectro, text_feature, visual_sr,
                   window_size, overlap_size, output_root, ID,
                   trans_included=False, spectro_trans=None, mel_spectro_trans=None):
    frame_size = window_size * visual_sr
    hop_size = (window_size - overlap_size) * visual_sr
    num_frame = get_num_frame(fkps_coor, frame_size, hop_size)
    text_frame_size = 10
    text_hop_size = get_text_hop_size(text_feature, text_frame_size, num_frame)

    # start sliding through and generating data
    if trans_included:
        assert spectro_trans is not None, "Transgender spectrogram should be provided"
        assert mel_spectro_trans is not None, "Transgender mel spectrogram should be provided"

        # expend facial keypoints by shifting along x/y/z-axis
        shifted_fc, shifted_fcc = random_shift_fkps(fkps_coor, fkps_coor_conf)

        for i in range(num_frame):
            # for normal data
            frame_sample_fc = visual_padding(fkps_coor[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_fcc = visual_padding(fkps_coor_conf[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_gc = visual_padding(gaze_coor[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_gcc = visual_padding(gaze_coor_conf[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_spec = audio_padding(spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_mspec = audio_padding(mel_spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_text = text_padding(text_feature[i * text_hop_size:i * text_hop_size + text_frame_size],
                                             text_frame_size)
            # for transgender data
            frame_sample_fc_trans = visual_padding(shifted_fc[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_fcc_trans = visual_padding(shifted_fcc[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_spec_trans = audio_padding(spectro_trans[:, i * hop_size:i * hop_size + frame_size],
                                                    frame_size)
            frame_sample_mspec_trans = audio_padding(mel_spectro_trans[:, i * hop_size:i * hop_size + frame_size],
                                                     frame_size)
            # this 3 should stay the same so just copy so that the total length would match
            frame_sample_gc_trans = frame_sample_gc
            frame_sample_gcc_trans = frame_sample_gcc
            frame_sample_text_trans = frame_sample_text

            # start storing
            np.save(os.path.join(output_root, 'facial_keypoints', 'only_coordinate', f'{ID}-{i:02}_kps.npy'),
                    frame_sample_fc)
            np.save(os.path.join(output_root, 'facial_keypoints', 'coordinate+confidence', f'{ID}-{i:02}_kps.npy'),
                    frame_sample_fcc)
            np.save(os.path.join(output_root, 'gaze_vectors', 'only_coordinate', f'{ID}-{i:02}_gaze.npy'),
                    frame_sample_gc)
            np.save(os.path.join(output_root, 'gaze_vectors', 'coordinate+confidence', f'{ID}-{i:02}_gaze.npy'),
                    frame_sample_gcc)
            np.save(os.path.join(output_root, 'audio', 'spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_spec)
            np.save(os.path.join(output_root, 'audio', 'mel-spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_mspec)
            np.save(os.path.join(output_root, 'text', 'sentence_embeddings', f'{ID}-{i:02}_text.npy'),
                    frame_sample_text)
            # start storing transgender data
            np.save(os.path.join(output_root, 'facial_keypoints', 'only_coordinate', f'{ID}-{i:02}_kps_trans.npy'),
                    frame_sample_fc_trans)
            np.save(
                os.path.join(output_root, 'facial_keypoints', 'coordinate+confidence', f'{ID}-{i:02}_kps_trans.npy'),
                frame_sample_fcc_trans)
            np.save(os.path.join(output_root, 'gaze_vectors', 'only_coordinate', f'{ID}-{i:02}_gaze_trans.npy'),
                    frame_sample_gc_trans)
            np.save(os.path.join(output_root, 'gaze_vectors', 'coordinate+confidence', f'{ID}-{i:02}_gaze_trans.npy'),
                    frame_sample_gcc_trans)
            np.save(os.path.join(output_root, 'audio', 'spectrogram', f'{ID}-{i:02}_audio_trans.npy'),
                    frame_sample_spec_trans)
            np.save(os.path.join(output_root, 'audio', 'mel-spectrogram', f'{ID}-{i:02}_audio_trans.npy'),
                    frame_sample_mspec_trans)
            np.save(os.path.join(output_root, 'text', 'sentence_embeddings', f'{ID}-{i:02}_text_trans.npy'),
                    frame_sample_text_trans)

        return num_frame * 2

    else:
        for i in range(num_frame):
            frame_sample_fc = visual_padding(fkps_coor[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_fcc = visual_padding(fkps_coor_conf[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_gc = visual_padding(gaze_coor[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_gcc = visual_padding(gaze_coor_conf[i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_spec = audio_padding(spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_mspec = audio_padding(mel_spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
            frame_sample_text = text_padding(text_feature[i * text_hop_size:i * text_hop_size + text_frame_size],
                                             text_frame_size)

            # start storing
            np.save(os.path.join(output_root, 'facial_keypoints', 'only_coordinate', f'{ID}-{i:02}_kps.npy'),
                    frame_sample_fc)
            np.save(os.path.join(output_root, 'facial_keypoints', 'coordinate+confidence', f'{ID}-{i:02}_kps.npy'),
                    frame_sample_fcc)
            np.save(os.path.join(output_root, 'gaze_vectors', 'only_coordinate', f'{ID}-{i:02}_gaze.npy'),
                    frame_sample_gc)
            np.save(os.path.join(output_root, 'gaze_vectors', 'coordinate+confidence', f'{ID}-{i:02}_gaze.npy'),
                    frame_sample_gcc)
            np.save(os.path.join(output_root, 'audio', 'spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_spec)
            np.save(os.path.join(output_root, 'audio', 'mel-spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_mspec)
            np.save(os.path.join(output_root, 'text', 'sentence_embeddings', f'{ID}-{i:02}_text.npy'),
                    frame_sample_text)

        return num_frame


if __name__ == '__main__':
    dele_ID = [300, 342, 373, 394, 398, 402, 416, 417, 444, 451, 458, 460, 480, 420]
    test_ID = [300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 373, 378, 384, 387,
               396, 399, 405, 407, 408, 410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462, 465, 466,
               467, 469, 470, 480, 481]
    train_ID = [302, 303, 304, 305, 307, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328,
                330, 331, 333, 335, 336, 338, 339, 340, 341, 343, 344, 345, 346, 347, 348, 350, 351, 352, 353, 355, 356,
                357, 358, 360, 362, 363, 364, 366, 367, 368, 369, 370, 371, 372, 374, 375, 376, 377, 379, 380, 381, 382,
                383, 385, 386, 388, 389, 390, 391, 392, 393, 395, 397, 400, 401, 402, 403, 404, 406, 409, 412, 413, 414,
                415, 416, 417, 418, 419, 420, 422, 423, 425, 426,
                427, 428, 429, 430, 433, 434, 436, 437, 439, 440, 441, 443, 444, 445, 446, 447, 448, 449, 451, 454, 455,
                456, 457, 458, 459, 463, 464, 468, 471, 472, 473, 474, 475, 476, 477,
                478, 479, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492]
    for ID in train_ID:

        # output root
        root = '.'
        root_dir = os.path.join(root, 'DAIC_WOZ-generated_database_exptrain', '{}'.format(ID), 'train')
        create_folders(root_dir)
        np.random.seed(1)

        # read  gt file
        gt_path = './Depression_Classification_work/database_generation/full_train_split_Depression_AVEC2017.csv'
        gt_df = pd.read_csv(gt_path)

        # initialization
        sent2vec = SentenceTransformer('all-mpnet-base-v2')
        window_size = 60  # 60s
        overlap_size = 10  # 10s
        GT = {'original_data':
                  {'no_gender_balance':
                       {'ID_gt': [], 'gender_gt': [], 'binary_gt': [], 'score_gt': [], 'subscores_gt': []},
                   'gender_balance':
                       {'ID_gt': [], 'gender_gt': [], 'binary_gt': [], 'score_gt': [], 'subscores_gt': []}},
              'clipped_data':
                  {'no_gender_balance':
                       {'ID_gt': [], 'gender_gt': [], 'binary_gt': [], 'score_gt': [], 'subscores_gt': []},
                   'gender_balance':
                       {'ID_gt': [], 'gender_gt': [], 'binary_gt': [], 'score_gt': [], 'subscores_gt': []}},
              'key_data':
                  {'no_gender_balance':
                       {'ID_gt': [], 'gender_gt': [], 'binary_gt': [], 'score_gt': [], 'subscores_gt': []},
                   'gender_balance':
                       {'ID_gt': [], 'gender_gt': [], 'binary_gt': [], 'score_gt': [], 'subscores_gt': []}}
              }

        for i in range(len(gt_df)):
            # extract training gt details
            patient_ID = gt_df['Participant_ID'][i]
            if patient_ID in dele_ID:
                continue
            if patient_ID != ID:
                continue
            binary_gt = gt_df['PHQ8_Binary'][i]
            score_gt = gt_df['PHQ8_Score'][i]
            gender_gt = gt_df['Gender'][i]
            subscores_gt = gt_df.iloc[i, 4:].to_numpy().tolist()
            print(f'Processing Participant {patient_ID}, Gender: {gender_gt} ...')
            print(f'- PHQ Binary: {binary_gt}, PHQ Score: {score_gt}, Subscore: {subscores_gt}')

            # get all files path of participant
            keypoints_path = f'./DAIC_Woz/{patient_ID}_P/{patient_ID}_CLNF_features3D.txt'
            gaze_path = f'./DAIC_Woz/{patient_ID}_P/{patient_ID}_CLNF_gaze.txt'
            pose_path = f'./DAIC_Woz/{patient_ID}_P/{patient_ID}_CLNF_pose.txt'
            audio_path = f'./DAIC_Woz/{patient_ID}_P/{patient_ID}_AUDIO.wav'
            text_path = f'./DAIC_Woz/{patient_ID}_P/{patient_ID}_TRANSCRIPT.csv'
            audio_trans_path = f'./DAIC_Woz/{patient_ID}_P/output.wav'

            # read transcipt file
            text_df = pd.read_csv(text_path, sep='\t').fillna('')
            first_start_time = text_df['start_time'][0]
            last_stop_time = text_df['stop_time'][len(text_df) - 1]

            # read & process visual files
            gaze_coor, gaze_coor_conf = load_gaze(gaze_path)
            fkps_coor, fkps_coor_conf = load_keypoints(keypoints_path)
            visual_sr = 30  # 30Hz

            # read audio file
            audio, audio_sr = load_audio(audio_path)
            audio_trans, audio_trans_sr = load_audio(audio_trans_path)

            # extract text feature
            text_feature = sentence_embedding(text_df, model=sent2vec)

            ########################################
            # feature extraction for original_data #
            ########################################

            print(f'Extracting feature of Participant {patient_ID} for original_data...')

            # visual
            filtered_fkps_coor = fkps_coor[int(first_start_time * visual_sr):int(last_stop_time * visual_sr)]
            filtered_fkps_coor_conf = fkps_coor_conf[int(first_start_time * visual_sr):int(last_stop_time * visual_sr)]
            filtered_gaze_coor = gaze_coor[int(first_start_time * visual_sr):int(last_stop_time * visual_sr)]
            filtered_gaze_coor_conf = gaze_coor_conf[int(first_start_time * visual_sr):int(last_stop_time * visual_sr)]

            # audio
            filtered_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=True)
            filtered_audio_trans = audio_clipping(audio_trans, audio_trans_sr, text_df, zero_padding=True)
            # spectrogram, mel spectrogram
            spectro = normalize(convert_spectrogram(filtered_audio, frame_size=2048, hop_size=533))
            mel_spectro = normalize(convert_mel_spectrogram(filtered_audio, audio_sr,
                                                            frame_size=2048, hop_size=533, num_mel_bands=80))
            spectro_trans = normalize(convert_spectrogram(filtered_audio_trans, frame_size=2048, hop_size=533 * 3))
            mel_spectro_trans = normalize(convert_mel_spectrogram(filtered_audio_trans, audio_trans_sr,
                                                                  frame_size=2048, hop_size=533 * 3, num_mel_bands=80))

            ###################################################################
            # start creating data in 'original_data/no_gender_balance' folder #
            ###################################################################

            output_root = os.path.join(root_dir, 'original_data', 'no_gender_balance')
            num_frame = sliding_window(filtered_fkps_coor, filtered_fkps_coor_conf,
                                       filtered_gaze_coor, filtered_gaze_coor_conf,
                                       spectro, mel_spectro, text_feature, visual_sr,
                                       window_size, overlap_size, output_root, patient_ID,
                                       trans_included=False, spectro_trans=None, mel_spectro_trans=None)

            # replicate GT
            for _ in range(num_frame):
                GT['original_data']['no_gender_balance']['ID_gt'].append(patient_ID)
                GT['original_data']['no_gender_balance']['gender_gt'].append(gender_gt)
                GT['original_data']['no_gender_balance']['binary_gt'].append(binary_gt)
                GT['original_data']['no_gender_balance']['score_gt'].append(score_gt)
                GT['original_data']['no_gender_balance']['subscores_gt'].append(subscores_gt)

            ################################################################
            # start creating data in 'original_data/gender_balance' folder #
            ################################################################

            output_root = os.path.join(root_dir, 'original_data', 'gender_balance')
            num_frame = sliding_window(filtered_fkps_coor, filtered_fkps_coor_conf,
                                       filtered_gaze_coor, filtered_gaze_coor_conf,
                                       spectro, mel_spectro, text_feature, visual_sr,
                                       window_size, overlap_size, output_root, patient_ID,
                                       trans_included=True, spectro_trans=spectro_trans,
                                       mel_spectro_trans=mel_spectro_trans)

            # replicate GT
            for _ in range(num_frame):
                GT['original_data']['gender_balance']['ID_gt'].append(patient_ID)
                GT['original_data']['gender_balance']['gender_gt'].append(gender_gt)
                GT['original_data']['gender_balance']['binary_gt'].append(binary_gt)
                GT['original_data']['gender_balance']['score_gt'].append(score_gt)
                GT['original_data']['gender_balance']['subscores_gt'].append(subscores_gt)

            #######################################
            # feature extraction for clipped_data #
            #######################################

            print(f'Extracting feature of Participant {patient_ID} for clipped_data...')

            # visual
            clipped_fkps_coor = visual_clipping(fkps_coor, visual_sr, text_df)
            clipped_fkps_coor_conf = visual_clipping(fkps_coor_conf, visual_sr, text_df)
            clipped_gaze_coor = visual_clipping(gaze_coor, visual_sr, text_df)
            clipped_gaze_coor_conf = visual_clipping(gaze_coor_conf, visual_sr, text_df)

            # audio
            clipped_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=False)
            clipped_audio_trans = audio_clipping(audio_trans, audio_trans_sr, text_df, zero_padding=False)
            # spectrogram, mel spectrogram
            spectro = normalize(convert_spectrogram(clipped_audio, frame_size=2048, hop_size=533))
            mel_spectro = normalize(convert_mel_spectrogram(clipped_audio, audio_sr,
                                                            frame_size=2048, hop_size=533, num_mel_bands=80))
            spectro_trans = normalize(convert_spectrogram(clipped_audio_trans, frame_size=2048, hop_size=533 * 3))
            mel_spectro_trans = normalize(convert_mel_spectrogram(clipped_audio_trans, audio_trans_sr,
                                                                  frame_size=2048, hop_size=533 * 3, num_mel_bands=80))

            ##################################################################
            # start creating data in 'clipped_data/no_gender_balance' folder #
            ##################################################################

            output_root = os.path.join(root_dir, 'clipped_data', 'no_gender_balance')
            num_frame = sliding_window(clipped_fkps_coor, clipped_fkps_coor_conf,
                                       clipped_gaze_coor, clipped_gaze_coor_conf,
                                       spectro, mel_spectro, text_feature, visual_sr,
                                       window_size, overlap_size, output_root, patient_ID,
                                       trans_included=False, spectro_trans=None, mel_spectro_trans=None)

            # replicate GT
            for _ in range(num_frame):
                GT['clipped_data']['no_gender_balance']['ID_gt'].append(patient_ID)
                GT['clipped_data']['no_gender_balance']['gender_gt'].append(gender_gt)
                GT['clipped_data']['no_gender_balance']['binary_gt'].append(binary_gt)
                GT['clipped_data']['no_gender_balance']['score_gt'].append(score_gt)
                GT['clipped_data']['no_gender_balance']['subscores_gt'].append(subscores_gt)

            ###############################################################
            # start creating data in 'clipped_data/gender_balance' folder #
            ###############################################################

            output_root = os.path.join(root_dir, 'clipped_data', 'gender_balance')
            num_frame = sliding_window(clipped_fkps_coor, clipped_fkps_coor_conf,
                                       clipped_gaze_coor, clipped_gaze_coor_conf,
                                       spectro, mel_spectro, text_feature, visual_sr,
                                       window_size, overlap_size, output_root, patient_ID,
                                       trans_included=True, spectro_trans=spectro_trans,
                                       mel_spectro_trans=mel_spectro_trans)

            # replicate GT
            for _ in range(num_frame):
                GT['clipped_data']['gender_balance']['ID_gt'].append(patient_ID)
                GT['clipped_data']['gender_balance']['gender_gt'].append(gender_gt)
                GT['clipped_data']['gender_balance']['binary_gt'].append(binary_gt)
                GT['clipped_data']['gender_balance']['score_gt'].append(score_gt)
                GT['clipped_data']['gender_balance']['subscores_gt'].append(subscores_gt)

            #######################################
            # feature extraction for key_data #
            #######################################

            print(f'Extracting feature of Participant {patient_ID} for key_data...')

            # visual
            key_fkps_coor = visual_padding_key(visual_key(fkps_coor, visual_sr, text_df), visual_sr * window_size + 1)
            key_fkps_coor_conf = visual_padding_key(visual_key(fkps_coor_conf, visual_sr, text_df),
                                                    visual_sr * window_size + 1)
            key_gaze_coor = visual_padding_key(visual_key(gaze_coor, visual_sr, text_df), visual_sr * window_size + 1)
            key_gaze_coor_conf = visual_padding_key(visual_key(gaze_coor_conf, visual_sr, text_df),
                                                    visual_sr * window_size + 1)

            # audio
            key_audio = audio_key(audio, audio_sr, text_df, zero_padding=False)
            key_audio_trans = audio_key(audio_trans, audio_trans_sr, text_df, zero_padding=False)

            # spectrogram, mel spectrogram
            key_spectro = normalize(convert_spectrogram(key_audio, frame_size=2048, hop_size=533))
            key_mel_spectro = normalize(convert_mel_spectrogram(key_audio, audio_sr,
                                                                frame_size=2048, hop_size=533, num_mel_bands=80))
            key_spectro_trans = normalize(convert_spectrogram(key_audio_trans, frame_size=2048, hop_size=533 * 3))
            key_mel_spectro_trans = normalize(convert_mel_spectrogram(key_audio_trans, audio_trans_sr,
                                                                      frame_size=2048, hop_size=533 * 3,
                                                                      num_mel_bands=80))

            # extract text feature
            key_text_feature = text_padding_key(sentence_embedding(text_df, model=sent2vec), 10)
            ##################################################################
            # start creating data in 'key_data/no_gender_balance' folder #
            ##################################################################

            output_root = os.path.join(root_dir, 'key_data', 'no_gender_balance')
            num_frame = sliding_window(key_fkps_coor, key_fkps_coor_conf,
                                       key_gaze_coor, key_gaze_coor_conf,
                                       key_spectro, key_mel_spectro, key_text_feature, visual_sr,
                                       window_size, overlap_size, output_root, patient_ID,
                                       trans_included=False, spectro_trans=None, mel_spectro_trans=None)

            # replicate GT
            for _ in range(num_frame):
                GT['key_data']['no_gender_balance']['ID_gt'].append(patient_ID)
                GT['key_data']['no_gender_balance']['gender_gt'].append(gender_gt)
                GT['key_data']['no_gender_balance']['binary_gt'].append(binary_gt)
                GT['key_data']['no_gender_balance']['score_gt'].append(score_gt)
                GT['key_data']['no_gender_balance']['subscores_gt'].append(subscores_gt)

            ###############################################################
            # start creating data in 'key_data/gender_balance' folder #
            ###############################################################

            output_root = os.path.join(root_dir, 'key_data', 'gender_balance')
            num_frame = sliding_window(key_fkps_coor, key_fkps_coor_conf,
                                       key_gaze_coor, key_gaze_coor_conf,
                                       key_spectro, key_mel_spectro, key_text_feature, visual_sr,
                                       window_size, overlap_size, output_root, patient_ID,
                                       trans_included=True, spectro_trans=key_spectro_trans,
                                       mel_spectro_trans=key_mel_spectro_trans)

            # replicate GT
            for _ in range(num_frame):
                GT['key_data']['gender_balance']['ID_gt'].append(patient_ID)
                GT['key_data']['gender_balance']['gender_gt'].append(gender_gt)
                GT['key_data']['gender_balance']['binary_gt'].append(binary_gt)
                GT['key_data']['gender_balance']['score_gt'].append(score_gt)
                GT['key_data']['gender_balance']['subscores_gt'].append(subscores_gt)

            print(f'Participant {patient_ID} done!')
            print('=' * 50)

        # store new GT
        for k1, v1 in GT.items():
            for k2, v2 in v1.items():
                for k3, v3 in v2.items():
                    np.save(os.path.join(root_dir, k1, k2, f'{k3}.npy'), v3)

    print('All done!')
