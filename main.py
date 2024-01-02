
import time
import shutil
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from autolab_core import YamlConfig
import warnings
# local functions
from utils import *
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


def main(fold, dataloaders, visual_net, audio_net, text_net, evaluator, base_logger, writer, config, args, model_type,
         ckpt_path):
    model_parameters = [*visual_net.parameters()] + [*audio_net.parameters()] + [
        *text_net.parameters()] + [*evaluator.parameters()]
    optimizer, scheduler = get_optimizer_scheduler(model_parameters, config['OPTIMIZER'], config['SCHEDULER'])

    test_best_f1_score = 0
    test_epoch_best_f1 = 0
    test_best_acc = 0
    test_epoch_best_acc = 0
    acc, f1, pre, rec = [], [], [], []
    for epoch in range(config['EPOCHS']):
        log_and_print(base_logger,
                      f'Epoch: {epoch}  Current Best Test for f1-score: {test_best_f1_score} at epoch {test_epoch_best_f1}    fold {fold + 1}')
        log_and_print(base_logger,
                      f'Epoch: {epoch}  Current Best Test for accuracy: {test_best_acc} at epoch {test_epoch_best_acc}    fold {fold + 1}')

        for mode in ['train', 'test']:  # 'train',
            mode_start_time = time.time()

            score_gt = []
            subscores_gt = []
            binary_gt = []
            score_pred = []
            subscores_pred = []
            binary_pred = []

            if mode == 'train':
                visual_net.train()
                audio_net.train()
                text_net.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                visual_net.eval()
                audio_net.eval()
                text_net.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            total_loss = 0
            log_interval_loss = 0
            log_interval = 10
            batch_number = 0
            n_batches = len(dataloaders[mode])
            batches_start_time = time.time()

            for data in tqdm(dataloaders[mode]):
                batch_size = data['ID'].size(0)

                # store ground truth
                score_gt.extend(data['score_gt'].numpy().astype(float))  # 1D list
                binary_gt.extend(data['binary_gt'].numpy().astype(float))  # 1D list

                # TODO: extract features with multi-model ...
                # combine all models into a function

                gt_binary = data['binary_gt']
                temp_labels = [gt_binary - 0, gt_binary - 1]
                target_labels = []
                for i in range(2):
                    temp_target_labels = []
                    for j in range(temp_labels[0].size(0)):
                        if temp_labels[i][j] == 0:
                            temp_target_labels.append(j)
                    target_labels.append(torch.LongTensor(temp_target_labels[:]))
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].cuda()

                def model_processing(targe_labels, input):
                    # get facial visual feature with Deep Visual Net'
                    # # input shape for visual_net must be (B, C, F, T) = (batch_size, channels, features, time series)
                    # B, T, F, C = input['visual'].shape
                    # print('visual shape:', input['visual'].size())
                    # visual_input = input['visual'].permute(0, 3, 2, 1).contiguous()
                    # visual_features = visual_net(visual_input.to(args.device))  # output dim: [B, visual net output dim]

                    # get audio feature with Deep Audio Net'
                    # input shape for audio_net must be (B, F, T) = (batch_size, features, time series)
                    B, F, T = input['audio'].shape
                    # print('audio shape:', input['audio'].view(B, F, T).size())
                    audio_input = input['audio'].view(B, F, T)
                    audio_features = audio_net(audio_input.to(args.device))  # output dim: [B, audio net output dim]

                    # get Text features with Deep Text Net'
                    # input shape for text_net must be (B, F, T) = (batch_size, features, time series))
                    B, T, F = input['text'].shape
                    # print('text shape:', input['text'].permute(0, 2, 1).contiguous().size())
                    text_input = input['text'].permute(0, 2, 1).contiguous()
                    text_features = text_net(text_input.to(args.device))  # output dim: [B, text net output dim]

                    # ---------------------- Start evaluating with sub-attentional feature fusion ----------------------
                    # combine all features into shape: B, C=1, num_modal, audio net output dim
                    all_features = torch.stack([text_features, audio_features], dim=1).unsqueeze(
                        dim=1)  # visual_features,audio_features,text_features
                    # print('all_features shape:', all_features.size())
                    # get flops
                    # input_v = torch.randn(2, 1800, 72, 3).to(args.device)
                    probs = evaluator(all_features)

                    raw = probs[0]
                    raw = torch.stack(raw).permute(1, 0, 2).mean(dim=1)
                    l_pos_neg_self = torch.einsum('nc,ck->nk', [raw, raw.T])
                    l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
                    l_pos_neg_self = l_pos_neg_self.view(-1)

                    cl_self_labels = target_labels[gt_binary[0]]
                    for index in range(1, raw.size(0)):
                        cl_self_labels = torch.cat(
                            (cl_self_labels, target_labels[gt_binary[index]] + index * gt_binary.size(0)), 0)

                    l_pos_neg_self = l_pos_neg_self / 0.07
                    cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
                    cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)

                    return probs, cl_self_loss

                if mode == 'train':
                    gt = get_gt(data, config['EVALUATOR']['PREDICT_TYPE'])
                    if config['CRITERION']['USE_WEIGHTS']:
                        config['CRITERION']['WEIGHTS'] = get_crossentropy_weights(gt, config['EVALUATOR'])
                    criterion, mse_loss = get_criterion(config['CRITERION'], args)
                    probs, cle_loss = model_processing(target_labels, input=data)
                    loss = compute_loss(criterion, mse_loss, probs, cle_loss, gt, config['EVALUATOR'], args,
                                            use_soft_label=config['CRITERION']['USE_SOFT_LABEL'])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    # for test set, only do prediction
                    probs, cle_loss = model_processing(target_labels, input=data)

                # predict the final score
                pred_score = compute_score(probs[0], config['EVALUATOR'], args)
                score_pred.extend([pred_score[i].item() for i in range(batch_size)])  # 1D list
                binary_pred.extend(
                    [1 if pred_score[i].item() >= config['THRESHOLD'] else 0 for i in range(batch_size)])

                if mode == 'train':
                    # information per batch
                    total_loss += loss.item()
                    log_interval_loss += loss.item()
                    if batch_number % log_interval == 0 and batch_number > 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - batches_start_time) * 1000 / log_interval
                        current_loss = log_interval_loss / log_interval
                        print(f'| epoch {epoch:3d} | {mode} | {batch_number:3d}/{n_batches:3d} batches | '
                              f'LR {lr:7.6f} | ms/batch {ms_per_batch:5.2f} | loss {current_loss:8.5f} |')

                        # tensorboard
                        writer.add_scalar('Loss_per_{}_batches/{}'.format(log_interval, mode),
                                          current_loss, epoch * n_batches + batch_number)

                        log_interval_loss = 0
                        batches_start_time = time.time()
                else:
                    total_loss = np.nan

                batch_number += 1

            # information per mode
            print('Score prediction: {}'.format(score_pred[:40]))
            print('Score ground truth: {}'.format(score_gt[:40]))

            print('Binary prediction: {}'.format(binary_pred[:40]))
            print('Binary ground truth: {}'.format(binary_gt[:40]))

            average_loss = total_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            s_per_mode = time.time() - mode_start_time
            accuracy, correct_number = get_accuracy(binary_gt, binary_pred)

            # store information in logger and print
            print('-' * 110)
            msg = ('  End of {0}:\n  | time: {1:8.3f}s | LR: {2:7.6f} | Average Loss: {3:8.5f} | Accuracy: {4:5.2f}%'
                   ' ({5}/{6}) |').format(mode, s_per_mode, lr, average_loss, accuracy * 100, correct_number,
                                          len(binary_gt))
            log_and_print(base_logger, msg)
            print('-' * 110)

            # tensorboard
            writer.add_scalar('Loss_per_epoch/{}'.format(mode), average_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(mode), accuracy * 100, epoch)
            writer.add_scalar('Learning_rate/{}'.format(mode), lr, epoch)

            # Calculating additional evaluation scores
            log_and_print(base_logger, '  Output Scores:')

            # confusion matrix
            [[tp, fp], [fn, tn]] = standard_confusion_matrix(binary_gt, binary_pred)
            msg = (f'  - Confusion Matrix:\n'
                   '    -----------------------\n'
                   f'    | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                   '    -----------------------\n'
                   f'    | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                   '    -----------------------')
            log_and_print(base_logger, msg)

            # classification related
            tpr, tnr, precision, recall, f1_score = get_classification_scores(binary_gt, binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)
            if mode == 'test' and epoch >= 20:
                if np.isnan(f1_score):
                    f1_score = 0
                if np.isnan(precision):
                    precision = 0
                acc.append(accuracy)
                f1.append(f1_score)
                pre.append(precision)
                rec.append(recall)

            # regression related
            mae, mse, rmse, r2 = get_regression_scores(score_gt, score_pred)
            msg = ('  - Regression:\n'
                   '      MAE: {0:7.4f}\n'
                   '      MSE: {1:7.4f}\n'
                   '      RMSE: {2:7.4f}\n'
                   '      R2: {3:7.4f}\n').format(mae, mse, rmse, r2)
            log_and_print(base_logger, msg)

            # Calculate a Spearman correlation coefficien
            rho, p = stats.spearmanr(score_gt, score_pred)  # binary_gt, binary_pred
            msg = ('  - Correlation:\n'
                   '      Spearman correlation: {0:8.6f}\n').format(rho)
            log_and_print(base_logger, msg)

            # store the model score
            if mode == 'train':
                train_model_f1_score = f1_score
                train_model_acc = accuracy * 100
            elif mode == 'test':
                test_model_f1_score = f1_score
                test_model_acc = accuracy * 100

            # tensorboard
        if test_model_f1_score >= test_best_f1_score:
            test_best_f1_score = test_model_f1_score
            test_epoch_best_f1 = epoch

            msg = (f'--------- New best found for f1-score at epoch {epoch} !!! ---------\n'
                   f'- train score: {train_model_f1_score:8.6f}\n'
                   f'- test score: {test_model_f1_score:8.6f}\n'
                   f'--------- New best found for f1-score at epoch {epoch} !!! ---------\n')
            log_and_print(base_logger, msg)

            if args.save:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')
                file_path = os.path.join(ckpt_path,
                                         '{}_{}_f1_score-{:6.4f}.pt'.format(model_type, timestamp, test_best_f1_score))

                torch.save({'epoch': epoch,
                            'visual_net': visual_net.state_dict(),
                            'audio_net': audio_net.state_dict(),
                            'text_net': text_net.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_f1_score': test_best_f1_score},
                           file_path)

        if test_model_acc >= test_best_acc:
            test_best_acc = test_model_acc
            test_epoch_best_acc = epoch

            # score_pred = np.array(score_pred)
            # score_gt = np.array(score_gt)
            # score_p = pd.DataFrame(score_pred)
            # score_g = pd.DataFrame(score_gt)
            # writer_exc = pd.ExcelWriter(
            #         './ULCDL/excel_score/all_score_{}_ep{}_acc{}.xlsx'.format(fold, epoch, test_best_acc))
            # score_p.to_excel(writer_exc, 'page_1', float_format='%.5f')
            # score_g.to_excel(writer_exc, 'page_2', float_format='%.5f')
            # writer_exc.save()  # 关键4

            msg = (f'--------- New best found for accuracy at epoch {epoch} !!! ---------\n'
                   f'- train score: {train_model_acc:8.6f}\n'
                   f'- test score: {test_model_acc:8.6f}\n'
                   f'--------- New best found for accuracy at epoch {epoch} !!! ---------\n')
            log_and_print(base_logger, msg)

        scheduler.step()

    if args.save:
        best_model_weights_path = find_last_ckpts(ckpt_path, model_type, date=None)
        shutil.copy(best_model_weights_path, config['WEIGHTS']['PATH'])

    return sum(np.array(acc)) / (config['EPOCHS'] - 20), sum(np.array(f1)) / (config['EPOCHS'] - 20), sum(
        np.array(pre)) / (config[
                              'EPOCHS'] - 20), sum(np.array(rec)) / (config['EPOCHS'] - 20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        help="path to yaml file",
                        required=False,
                        default='config/config_main.yaml')
    parser.add_argument('--device',
                        type=str,
                        help="set up torch device: 'cpu' or 'cuda' (GPU)",
                        required=False,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # remember to set the gpu device number
    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        required=False,
                        default='2, 3')
    parser.add_argument('--save',
                        type=bool,
                        help='if set true, save the best model',
                        required=False,
                        default=False)
    args = parser.parse_args()

    # set up GPU
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load config file into dict() format
    config = YamlConfig(args.config_file)

    # create the output folder (name of experiment) for storing model result such as logger information
    if not os.path.exists(config['OUTPUT_DIR']):
        os.mkdir(config['OUTPUT_DIR'])
    # create the root folder for storing checkpoints during training
    if not os.path.exists(config['CKPTS_DIR']):
        os.mkdir(config['CKPTS_DIR'])
    # create the subfolder for storing checkpoints based on the model type
    if not os.path.exists(os.path.join(config['CKPTS_DIR'], config['TYPE'])):
        os.mkdir(os.path.join(config['CKPTS_DIR'], config['TYPE']))
    # create the folder for storing the best model after all epochs
    if not os.path.exists(config['MODEL']['WEIGHTS']['PATH']):
        os.mkdir(config['MODEL']['WEIGHTS']['PATH'])

    # print configuration
    print('=' * 40)
    print(config.file_contents)
    config.save(os.path.join(config['OUTPUT_DIR'], config['SAVE_CONFIG_NAME']))
    print('=' * 40)

    # initialize random seed for torch and numpy
    init_seed(config['MANUAL_SEED'])

    # get logger os.path.join(config['OUTPUT_DIR'], f'{config['TYPE']}_{config['LOG_TITLE']}.log')
    file_name = os.path.join(config['OUTPUT_DIR'], '{}.log'.format(config['TYPE']))
    base_logger = get_logger(file_name, config['LOG_TITLE'])
    # get summary writer for TensorBoard
    writer = SummaryWriter(os.path.join(config['OUTPUT_DIR'], 'runs'))

    m = 'train'
    if m == 'train':
        train_dataset = DepressionDataset(config['DATA'][f'TRAIN_ROOT_DIR'.upper()], 'train',
                                          use_mel_spectrogram=config['DATA']['USE_MEL_SPECTROGRAM'],
                                          visual_with_gaze=config['DATA']['VISUAL_WITH_GAZE'],
                                          transform=transforms.Compose([ToTensor(
                                              'train')]))  # Rescale(data_config['RESCALE_SIZE']), Padding(data_config['PADDING'])

        k = 3
        kf = KFold(n_splits=k, shuffle=True)
        Accuracy, F1_score, Precision, Recall = [], [], [], []
        for fold, (train_indices, test_indices) in enumerate(kf.split(train_dataset)):
            print(f'fold{fold + 1}/{k}')
            dataloaders = {}
            sampler_train = SubsetRandomSampler(train_indices)
            dataloaders['train'] = DataLoader(train_dataset,
                                              batch_size=config['DATA']['BATCH_SIZE'],
                                              num_workers=config['DATA']['NUM_WORKERS'],
                                              sampler=sampler_train,
                                              drop_last=True)
            sampler_test = SubsetRandomSampler(test_indices)
            dataloaders['test'] = DataLoader(train_dataset,
                                             batch_size=config['DATA']['BATCH_SIZE'],
                                             num_workers=config['DATA']['NUM_WORKERS'],
                                             sampler=sampler_test)

            ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
            model_type = config['TYPE']
            visual_net, audio_net, text_net, evaluator = get_models(config['MODEL'], args, model_type, ckpt_path)


            accuracy, f1_score, precision, recall = main(fold, dataloaders, visual_net, audio_net, text_net, evaluator,
                                                         base_logger, writer, config['MODEL'], args, model_type,
                                                         ckpt_path)
            Accuracy.append(accuracy)
            F1_score.append(f1_score)
            Precision.append(precision)
            Recall.append(recall)
            writer.close()

        print('F1_score:', sum(F1_score) / len(F1_score))
        print('Recall:', sum(Recall) / len(Recall))
        print('Precision:', sum(Precision) / len(Precision))
        print('Accuracy:', sum(Accuracy) / len(Accuracy))

    else:
        test_dataset = DepressionDataset(config['DATA'][f'TEST_ROOT_DIR'.upper()], 'test',
                                         use_mel_spectrogram=config['DATA']['USE_MEL_SPECTROGRAM'],
                                         visual_with_gaze=config['DATA']['VISUAL_WITH_GAZE'],
                                         transform=transforms.Compose([ToTensor(  # Padding(config['DATA']['PADDING']),
                                             'test')]))  # Rescale(data_config['RESCALE_SIZE']), Padding(data_config['PADDING']) + Augmentation TODO !!!
        dataloaders = {}
        dataloaders['test'] = DataLoader(test_dataset,
                                         batch_size=config['DATA']['BATCH_SIZE'],
                                         num_workers=config['DATA']['NUM_WORKERS'])

        fold = 0
        # get models
        ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
        model_type = config['TYPE']
        visual_net, audio_net, text_net, evaluator = get_models(config['MODEL'], args, model_type, ckpt_path)

        accuracy, f1_score, precision, recall = main(fold, dataloaders, visual_net, audio_net, text_net, evaluator,
                                                     base_logger, writer, config['MODEL'], args, model_type, ckpt_path)
