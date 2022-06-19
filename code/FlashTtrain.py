import os
import random
import warnings
import time
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, \
    f1_score, recall_score, precision_score, label_ranking_average_precision_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import MPNET_PRETRAINED_MODEL_ARCHIVE_LIST

from utils.data import *
from utils.model import myFlashTransformer
from utils.util import save_model, load_model, save_param

onnx_exec = True
warnings.filterwarnings("ignore")

N_EPOCHS = 15
kf = 5
num_tokens = 26
FLASHT_dim = 64
FLASHT_depth = 6
encoderConv = 16
kernel_size = 3
stride = 1
pool_dim = 20

conv_dim = 8
convPoolLen = 10

group_size = 3
l1 = 1
l2 = 20
l3 = 10
l4 = 2
dropout = 0.4
LR = 0.001
step_size = 5
batch_size = 32
gamma = 0.7
slice_loc = [1, 1000]
ratio = [0.8, 0.2]
multi_sample = 4
root = '/home/duanxf/model/antipep'  # 获取项目根目录
name = os.path.basename(__file__).split(".")[0]
title = datetime.datetime.now().strftime("%b%d_%Y_%H.%M")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TrainDataPath = '/home/duanxf/code/antipep/dataset/AMPDB/amp.csv'
# TestDataPath = 'dataset/Benchmark/sAMPs.csv'
trainFold, validFold = k_fold_split(all_peptide_file=TrainDataPath,
                                    kf_num=kf, slice_loc=slice_loc, )
trainFold_loader = [DataLoader(dataset, shuffle=True, batch_size=batch_size,
                               collate_fn=collate_function) for dataset in trainFold]
validFold_loader = [DataLoader(dataset, shuffle=True, batch_size=batch_size,
                               collate_fn=collate_function) for dataset in validFold]


def score2label(predictions):
    predictions_list = []
    for i in range(len(predictions)):
        if predictions[i][0] >= predictions[i][1]:
            predictions_list.append(np.array([1, 0]))
        else:
            predictions_list.append(np.array([0, 1]))
    return predictions_list
    


def catFeature(batch):
    seq_label = [torch.tensor(i).unsqueeze(0).to(device) for i in batch[0]]
    label_batch = [torch.tensor(i, dtype=torch.float).unsqueeze(0).to(device)
                   for i in label2OneHot(batch[1])]
    label_batch = torch.cat(label_batch, 0)
    return seq_label, label_batch


def metrics_epoch(label, pre):
    dict_mcc = {'1': '-1', '0': '1'}
    acc = accuracy_score(
        label, pre)
    recall = recall_score(
        label, pre,  average='macro')
    f1 = f1_score(
        label, pre, average='macro')
    precision = precision_score(
        label, pre, average='macro')
    mcc = matthews_corrcoef(
        [int(dict_mcc[str(int(i[0]))]) for i in label], [
            int(dict_mcc[str(int(i[0]))]) for i in pre]
    )
    return acc, recall, f1, precision, mcc


def train(train_model, iterator, optimizer, criterion,):
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_mcc = 0
    train_model.train()
    prediction = []
    label = []
    for batch in tqdm(iterator, ncols=0):
        optimizer.zero_grad()
        seq, label_batch = catFeature(batch)
        m_pred = train_model(seq)
        losses = 0
        for i in m_pred:
            loss = criterion(i, label_batch)
            losses += loss
        loss = losses/multi_sample
        loss.backward()
        optimizer.step()
        prediction_batch = m_pred[random.randint(0, multi_sample-1)].cpu().data.numpy()
        label_batch = list(label_batch.cpu().data.numpy())
        prediction_label = score2label(
            list(prediction_batch))
        acc, recall, f1, precision, mcc = metrics_epoch(
            label_batch, prediction_label)
        epoch_loss += np.float64(loss)
        epoch_acc += np.float64(acc)
        epoch_recall += np.float64(recall)
        epoch_f1 += np.float64(f1)
        epoch_precision += np.float64(precision)
        epoch_mcc += np.float64(mcc)
        prediction = prediction + list(prediction_batch)
        label = label + label_batch
        len_iter = len(iterator)
    return epoch_loss / len_iter,\
        epoch_acc / len_iter, \
        epoch_precision / len_iter,\
        epoch_recall / len_iter, \
        epoch_f1 / len_iter,\
        epoch_mcc / len_iter, \
        roc_auc_score(label, prediction), \
        label_ranking_average_precision_score(label, prediction)


def valid(valid_model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_mcc = 0
    valid_model.eval()
    prediction = []
    label = []
    for batch in iterator:
        seq, label_batch = catFeature(batch)
        m_pred = valid_model(seq)
        losses = 0
        for i in m_pred:
            loss = criterion(i, label_batch)
            losses += loss
        loss = losses/multi_sample
        prediction_batch = m_pred[random.randint(0, multi_sample-1)].cpu().data.numpy()
        label_batch = list(label_batch.cpu().data.numpy())
        prediction_label = score2label(list(prediction_batch))
        acc, recall, f1, precision, mcc = metrics_epoch(
            label_batch, prediction_label)
        epoch_loss += np.float64(loss)
        epoch_acc += np.float64(acc)
        epoch_recall += np.float64(recall)
        epoch_f1 += np.float64(f1)
        epoch_precision += np.float64(precision)
        epoch_mcc += np.float64(mcc)
        prediction = prediction + list(prediction_batch)
        label = label + label_batch
        len_iter = len(iterator)
    return epoch_loss / len_iter, epoch_acc / len_iter, \
        epoch_precision / len_iter, epoch_recall / len_iter, \
        epoch_f1 / len_iter, epoch_mcc / len_iter, roc_auc_score(label, prediction), \
        label_ranking_average_precision_score(label, prediction)


ave_list = [0, 0, 0, 0, 0, 0, 0, 0]
for i in tqdm(range(kf), ncols=0):
    model = myFlashTransformer(multi_sample=multi_sample,
                               num_tokens=num_tokens,
                               dropout=dropout,
                               pool_dim=pool_dim,
                               kernel_size=kernel_size,
                               stride=stride,
                               conv_dim=conv_dim,
                               encoderConv=encoderConv,
                               convPoolLen=convPoolLen,
                               FLASHT_dim=FLASHT_dim,
                               FLASHT_depth=FLASHT_depth,
                               group_size=group_size,
                               l1=l1, l2=l2, l3=l3, l4=l4)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # criterion = nn.BCELoss().to(device)
    criterion = nn.BCELoss().to(device)

    best_valid_mcc = -1.0
    train_loader = trainFold_loader[i]
    valid_loader = validFold_loader[i]
    tqdm.write("train loader {}/{} batch/size valid loader {}/{} batch/size ".format(
        len(train_loader), batch_size, len(valid_loader), batch_size))
    for epoch in tqdm(range(N_EPOCHS), ncols=0):
        start_time = time.time()
        train_loss, train_acc, train_precision, train_recall, train_f1, train_mcc, train_roc, train_auc = train(
            model, train_loader, optimizer, criterion)
        scheduler.step()
        memory = torch.cuda.memory_allocated()/(1024*1024*8)
        res = 'kf {}/{} epoch {}/{} TRAIN: loss: {:.4f} acc: {:.4f} recall: {:.4f} f1: {:.4f} precision:{:.4f} mcc:{:.4f} roc: {:.4f} auc: {:.4f} LR:{:.5f} memory: {:.8f}'.format(
            i+1, kf, epoch+1, N_EPOCHS, train_loss, train_acc, train_recall, train_f1, train_precision, train_mcc, train_roc, train_auc, scheduler.get_lr()[0], memory)
        tqdm.write(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S"))
        tqdm.write(res)

        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_mcc, valid_roc, valid_auc = valid(
            model, valid_loader, criterion)

        if valid_mcc > best_valid_mcc:
            best_valid_mcc = valid_mcc
            save_model(root=root, model=model, p=[valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_mcc, valid_roc, valid_auc],
                       name=name, k=i, title=title)
        res = 'kf {}/{} epoch {}/{} valid: loss: {:.4f} acc: {:.4f} recall: {:.4f} f1: {:.4f} precision:{:.4f} mcc:{:.4f} roc: {:.4f} auc: {:.4f}'.format(
            i+1, kf, epoch+1, N_EPOCHS, valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_mcc, valid_roc, valid_auc)
        tqdm.write(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S"))
        tqdm.write(res)

    model_valid = load_model(root=root, name=name, k=i, title=title)
    model_valid = model_valid.to(device=device)
    valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_mcc, valid_roc, valid_auc = valid(
        model_valid, valid_loader, criterion)
    res = '{} FOLD BEST valid: loss: {:.4f} acc: {:.4f} recall: {:.4f} f1: {:.4f} precision:{:.4f} mcc:{:.4f} roc: {:.4f} auc: {:.4f}'.format(
        i+1, valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_mcc, valid_roc, valid_auc)
    tqdm.write('----------------------------------------')
    tqdm.write(res)
    tqdm.write('----------------------------------------')
    ave_list = [i + j for i, j in zip(ave_list, [valid_loss, valid_acc, valid_precision,
                                                 valid_recall, valid_f1, valid_mcc, valid_roc, valid_auc])]
save_param(root, name, title, ave_list+[kf])
