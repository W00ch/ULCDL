from itertools import count
import os
import sys
import random
import logging
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms

# local functions
from dataset.dataset import DepressionDataset, Padding, Rescale, RandomCrop, ToTensor
from models.network import CPLSTM_Visual, CPLSTM_Audio, CPLSTM_Text
from models.evaluator import Evaluator
from models.fusion import Bottleneck


def init_seed(manual_seed):
    """
    Set random seed for torch and numpy.
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filepath, log_title):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 54 + log_title + '-' * 54)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)




def get_sampler_score(score_gt):
    class_sample_ID, class_sample_count = np.unique(score_gt, return_counts=True)
    weight = 1. / class_sample_count
    samples_weight = np.zeros(score_gt.shape)
    for i, sample_id in enumerate(class_sample_ID):
        indices = np.where(score_gt == sample_id)[0]
        value = weight[i]
        samples_weight[indices] = value
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_dataloaders(data_config):

    dataloaders = {}
    for mode in ['train', 'test']:
        if mode == 'test':
            # for test dataset, we don't need shuffle, sampler and augmentation
            dataset = DepressionDataset(data_config[f'{mode}_ROOT_DIR'.upper()], mode,
                                        use_mel_spectrogram=data_config['USE_MEL_SPECTROGRAM'],
                                        visual_with_gaze=data_config['VISUAL_WITH_GAZE'],
                                        transform=transforms.Compose([ToTensor(mode)]))
            # sampler = get_sampler_score(dataset.score_gt)
            dataloaders[mode] = DataLoader(dataset,
                                           batch_size=data_config['BATCH_SIZE'],
                                           num_workers=data_config['NUM_WORKERS'],
                                           # sampler=sampler,
                                           shuffle=True,
                                           drop_last=True)

        else:
            dataset = DepressionDataset(data_config[f'{mode}_ROOT_DIR'.upper()], mode,
                                        use_mel_spectrogram=data_config['USE_MEL_SPECTROGRAM'],
                                        visual_with_gaze=data_config['VISUAL_WITH_GAZE'],
                                        transform=transforms.Compose([ToTensor(mode)]))  # Rescale(data_config['RESCALE_SIZE']), Padding(data_config['PADDING']) + Augmentation TODO

            sampler = get_sampler_score(dataset.score_gt)
            dataloaders[mode] = DataLoader(dataset,
                                           batch_size=data_config['BATCH_SIZE'],
                                           num_workers=data_config['NUM_WORKERS'], 
                                           sampler=sampler,
                                           # shuffle=True,
                                           drop_last=True)

    return dataloaders


def find_last_ckpts(path, key, date=None):
    ckpts = list(sorted(os.listdir(path)))

    if date is not None:
        # match the date format
        date_format = "%Y-%m-%d"
        try:
            datetime.strptime(date, date_format)
            matched = True
        except ValueError:
            matched = False
        assert matched, "The given date is the incorrect date string format. It should be YYYY-MM-DD"

        key = '{}_{}'.format(key, date)
    else:
        key = str(key)

    # filter the files
    ckpts = list(filter(lambda f: f.startswith(key), ckpts))
    # get whole file path
    last_ckpt = os.path.join(path, ckpts[-1])

    return last_ckpt


def get_models(model_config, args, model_type=None, ckpt_path=None):
    """
    Get the different deep model net as encoder backbone and the evaluator with parameters moved to GPU.
    """
    visual_net = CPLSTM_Visual(input_dim=model_config['VISUAL_NET']['INPUT_DIM'],
                                 output_dim=model_config['VISUAL_NET']['OUTPUT_DIM'], 
                                 conv_hidden=model_config['VISUAL_NET']['CONV_HIDDEN'], 
                                 lstm_hidden=model_config['VISUAL_NET']['LSTM_HIDDEN'],
                                 num_layers=model_config['VISUAL_NET']['NUM_LAYERS'], 
                                 activation=model_config['VISUAL_NET']['ACTIVATION'],
                                 norm = model_config['VISUAL_NET']['NORM'], 
                                 dropout=model_config['VISUAL_NET']['DROPOUT'])

    audio_net = CPLSTM_Audio(input_dim=model_config['AUDIO_NET']['INPUT_DIM'],
                               output_dim=model_config['AUDIO_NET']['OUTPUT_DIM'], 
                               conv_hidden=model_config['AUDIO_NET']['CONV_HIDDEN'], 
                               lstm_hidden=model_config['AUDIO_NET']['LSTM_HIDDEN'],
                               num_layers=model_config['AUDIO_NET']['NUM_LAYERS'], 
                               activation=model_config['AUDIO_NET']['ACTIVATION'],
                               norm = model_config['AUDIO_NET']['NORM'], 
                               dropout=model_config['AUDIO_NET']['DROPOUT'])

    text_net = CPLSTM_Text(input_dim=model_config['TEXT_NET']['INPUT_DIM'],
                             output_dim=model_config['TEXT_NET']['OUTPUT_DIM'], 
                             conv_hidden=model_config['TEXT_NET']['CONV_HIDDEN'], 
                             lstm_hidden=model_config['TEXT_NET']['LSTM_HIDDEN'],
                             num_layers=model_config['TEXT_NET']['NUM_LAYERS'], 
                             activation=model_config['TEXT_NET']['ACTIVATION'],
                             norm = model_config['TEXT_NET']['NORM'], 
                             dropout=model_config['TEXT_NET']['DROPOUT'])

    evaluator = Evaluator(feature_dim=model_config['EVALUATOR']['INPUT_FEATURE_DIM'],
                          output_dim=model_config['EVALUATOR']['CLASSES_RESOLUTION'],
                          predict_type=model_config['EVALUATOR']['PREDICT_TYPE'],
                          num_subscores=model_config['EVALUATOR']['N_SUBSCORES'],
                          dropout=model_config['EVALUATOR']['DROPOUT'],
                          attention_config=model_config['EVALUATOR']['ATTENTION'])

    # move to GPU
    visual_net = visual_net.to(args.device)
    audio_net = audio_net.to(args.device)
    text_net = text_net.to(args.device)
    evaluator = evaluator.to(args.device)

    # load model weights
    # if weights_path is not None:
    #     model_config['WEIGHTS']['INCLUDED'] = [x.lower() for x in model_config['WEIGHTS']['INCLUDED']]
    #
    #     checkpoint = torch.load(weights_path)
    #
    #     if 'visual_net' in model_config['WEIGHTS']['INCLUDED']:
    #         print("Loading Deep Visual Net weights from {}".format(weights_path))
    #         visual_net.load_state_dict(checkpoint['visual_net'])
    #
    #     if 'audio_net' in model_config['WEIGHTS']['INCLUDED']:
    #         print("Loading Deep Audio Net weights from {}".format(weights_path))
    #         audio_net.load_state_dict(checkpoint['audio_net'])
    #
    #     if 'text_net' in model_config['WEIGHTS']['INCLUDED']:
    #         print("Loading Deep Text Net weights from {}".format(weights_path))
    #         text_net.load_state_dict(checkpoint['text_net'])

        # if 'fusion_net' in model_config['WEIGHTS']['INCLUDED']:
        #     print("Loading Attention Fusion Layer weights from {}".format(weights_path))
        #     fusion_net.load_state_dict(checkpoint['fusion_net'])

        # if 'evaluator' in model_config['WEIGHTS']['INCLUDED']:
        #     print("Loading MUSDL weights from {}".format(weights_path))
        #     evaluator.load_state_dict(checkpoint['evaluator'])

    return visual_net, audio_net, text_net, evaluator  # fusion_net


def get_crossentropy_weights_whole_data(data_config, evaluator_config):

    root_dir = data_config['{}_ROOT_DIR'.format(data_config['MODE']).upper()]

    if evaluator_config['PREDICT_TYPE'] == 'subscores':
        gt_path = os.path.join(root_dir, 'subscores_gt.npy')
        gt = np.load(gt_path)

        weights = np.zeros(evaluator_config['N_CLASSES'])
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[labels[i]] = 1. / counts[i]

    elif evaluator_config['PREDICT_TYPE'] == 'score':
        gt_path = os.path.join(root_dir, 'score_gt.npy')
        gt = np.load(gt_path)

        weights = np.zeros(evaluator_config['N_CLASSES'])
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[labels[i]] = 1. / counts[i]

    elif evaluator_config['PREDICT_TYPE'] == 'binary':
        gt_path = os.path.join(root_dir, 'binary_gt.npy')
        gt = np.load(gt_path)

        weights = np.zeros(evaluator_config['N_CLASSES'])
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[labels[i]] = 1. / counts[i]
    
    else:
        raise AssertionError("Unknown 'PREDICT_TYPE' for evaluator!", evaluator_config['PREDICT_TYPE'])

    return weights


def get_crossentropy_weights(gt, evaluator_config):

    if evaluator_config['PREDICT_TYPE'] == 'subscores':

        weights = np.zeros(evaluator_config['N_CLASSES'])
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[int(labels[i])] = 1. / counts[i]

    elif evaluator_config['PREDICT_TYPE'] == 'score':

        weights = np.zeros(evaluator_config['N_CLASSES'])
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[int(labels[i])] = 1. / counts[i]

    elif evaluator_config['PREDICT_TYPE'] == 'binary':

        weights = np.zeros(evaluator_config['N_CLASSES'])
        labels, counts = np.unique(gt, return_counts=True)
        for i in range(len(labels)):
            weights[int(labels[i])] = 1. / counts[i]
    
    else:
        raise AssertionError("Unknown 'PREDICT_TYPE' for evaluator!", evaluator_config['PREDICT_TYPE'])

    return weights


def get_criterion(criterion_config, args):

    if criterion_config['USE_SOFT_LABEL']:

        criterion = nn.KLDivLoss()
        mse_loss = nn.MSELoss()

        return criterion, mse_loss
    else:
        if criterion_config['USE_WEIGHTS']:
            weights = torch.tensor(criterion_config['WEIGHTS']).type(torch.FloatTensor).to(args.device)
            a = criterion_config['WEIGHTS']
            criterion = nn.CrossEntropyLoss(weight=weights)
        
        else:
            
            criterion = nn.CrossEntropyLoss()

        return criterion


def get_optimizer_scheduler(model_parameters, optimizer_config, scheduler_config):
    # get optimizer and scheduler
    optimizer = torch.optim.Adam(model_parameters, betas=(0.9, 0.999),
                                     lr=optimizer_config['LR'],
                                     weight_decay=optimizer_config['WEIGHT_DECAY'])
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=scheduler_config['STEP_SIZE'],
                                                    gamma=scheduler_config['GAMMA'])
    
    return optimizer, scheduler


def get_gt(data, predict_type):
    if predict_type == 'subscores':
        gt = data['subscores_gt']

    elif predict_type == 'score':
        gt = data['score_gt']

    elif predict_type == 'binary':
        gt = data['binary_gt']
    
    else:
        raise AssertionError("Unknown 'PREDICT_TYPE' for evaluator!", predict_type)

    return gt


def compute_score(probs, evaluator_config, args):

    if evaluator_config['PREDICT_TYPE'] == 'subscores':
        factor = evaluator_config['N_CLASSES'] / evaluator_config['CLASSES_RESOLUTION']
        subscores_pred = torch.stack([prob.argmax(dim=-1) * factor
                                     for prob in probs], dim=1).sort()[0].to(int).to(float)  # (number of batch, num_subscores)

        score_pred = torch.sum(subscores_pred, dim=1)  # number of batch x 1

        return score_pred.to(args.device)
    
    else:
        factor = evaluator_config['N_CLASSES'] / evaluator_config['CLASSES_RESOLUTION']
        score_pred = (probs.argmax(dim=-1) * factor).to(int).to(float)

        return score_pred.to(args.device)



def convert_soft_gt(gt, evaluator_config):

    if evaluator_config['PREDICT_TYPE'] == 'subscores':
        factor = (evaluator_config['N_CLASSES'] - 1) / (evaluator_config['CLASSES_RESOLUTION'] - 1)
        tmp = [stats.norm.pdf(np.arange(evaluator_config['CLASSES_RESOLUTION']), loc=score / factor,
                              scale=evaluator_config['STD']).astype(np.float32) for score in gt]

        tmp = np.stack(tmp)  # shape: (num_subscores, class_resolution)
    
    else:
        factor = (evaluator_config['N_CLASSES'] - 1) / (evaluator_config['CLASSES_RESOLUTION'] - 1)
        tmp = stats.norm.pdf(np.arange(evaluator_config['CLASSES_RESOLUTION']), loc=gt / factor,
                             scale=evaluator_config['STD']).astype(np.float32) # shape: (class_resolution, )

    return torch.from_numpy(tmp / tmp.sum(axis=-1, keepdims=True))


def get_soft_gt(gt, evaluator_config):

    soft_gt = torch.tensor([[]])

    # iterate through each batch 
    for i in range(len(gt)):

        current_gt = gt[i]
        converted_current_gt = convert_soft_gt(current_gt, evaluator_config)
        if i == 0:
            soft_gt = converted_current_gt.unsqueeze(dim=0)
        else:
            soft_gt = torch.cat([soft_gt, converted_current_gt.unsqueeze(dim=0)], dim=0)

    return soft_gt


def compute_loss(criterion, mse_loss, probs, cle_loss, gt, evaluator_config, args, use_soft_label=False):
    a = 0.8

    if use_soft_label:
        soft_gt = get_soft_gt(gt, evaluator_config)
        score_gt = gt.sum(dim=1, keepdim=True)

        if evaluator_config['PREDICT_TYPE'] == 'subscores':
            loss_kl = sum([criterion(torch.log(probs[0][i]), soft_gt[:, i].to(args.device))
                        for i in range(evaluator_config['N_SUBSCORES'])])
            loss_mse = mse_loss(probs[1].to(args.device), score_gt.to(args.device))
            loss = a * loss_kl + (1 - a) * loss_mse * 0.01 + cle_loss * 0.004
        else:
            loss = criterion(torch.log(probs), soft_gt.to(args.device))

    else:

        if evaluator_config['PREDICT_TYPE'] == 'subscores':
            pred_prob = torch.stack([prob for prob in probs], dim=1)
            loss = criterion(pred_prob.permute(0, 2, 1).contiguous(),
                             gt.type(torch.LongTensor).to(args.device))
        else:
            loss = criterion(probs, gt.type(torch.LongTensor).to(args.device))
            
    return loss


def standard_confusion_matrix(gt, pred):
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
    [[tn, fp], [fn, tp]] = metrics.confusion_matrix(np.asarray(gt), np.asarray(pred))
    return np.array([[tp, fp], [fn, tn]])


def get_accuracy(gt, pred):
    [[tp, fp], [fn, tn]] = standard_confusion_matrix(gt, pred)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    correct_number = tp + tn
    return accuracy, correct_number


def get_classification_scores(gt, pred):
    [[tp, fp], [fn, tn]] = standard_confusion_matrix(gt, pred)
    # TPR(sensitivity), TNR(specificity)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    # Precision, Recall, F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return tpr, tnr, precision, recall, f1_score


def get_regression_scores(gt, pred):
    gt = np.array(gt).astype(float)
    pred = np.array(pred).astype(float)
    mae = metrics.mean_absolute_error(gt, pred)  # mean absolute error
    mse = metrics.mean_squared_error(gt, pred)   # mean square error
    rmse = np.sqrt(mse)  # or mse**(0.5)         # root mean square error
    r2 = metrics.r2_score(gt, pred)              # Coefficient of determination
    return mae, mse, rmse, r2


