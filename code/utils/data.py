import datetime
from random import shuffle

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch.utils.data import Dataset


def label_encoder(str_seq_list):
    dict_seq = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
                'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
                'Z', 'X', 'C', 'V', 'B', 'N', 'M', ]
    le = preprocessing.LabelEncoder()
    le.fit(dict_seq)
    return [le.transform(list(str_seq.upper())) for str_seq in str_seq_list]


def label2OneHot(label_list):
    dict_label = {
        0: np.array([1, 0]),
        1: np.array([0, 1])
    }
    label_OneHot = []
    for i in range(len(label_list)):
        label_OneHot.append(np.array(dict_label[label_list[i]]))
    return label_OneHot


class AMPDataset(Dataset):
    def __init__(self, seq, label):
        self.seq = label_encoder(seq)
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        data = []
        seq = self.seq[item]
        label = self.label[item]
        data.append(seq)
        data.append(label)
        return data


def collate_function(data_list):
    real_list = []
    label = []
    seq_label = []
    for item in data_list:
        seq_label.append(item[0])
        label.append(item[1])
    real_list.append(seq_label)
    real_list.append(label)
    return real_list


def k_fold_split_with_test(all_peptide_file, TestDataPath, kf_num, slice_loc):
    all_peptide = pd.read_csv(all_peptide_file, header=0, names=['seq', 'interaction'])
    testDf = pd.read_csv(TestDataPath, header=0, names=['seq', 'interaction'])
    testDf = shuffle(testDf)
    test_interaction = testDf['interaction'].to_list()
    test_seq = testDf['seq'].to_list()
    testset = AMPDataset(test_seq, test_interaction)
    all_peptide = shuffle(all_peptide)
    
    if slice_loc[0] == 1:
        all_peptide = all_peptide[0:slice_loc[1]]
    print(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S"),
          "load dataset ", len(all_peptide), ' lenth')
    kf = KFold(kf_num, shuffle=True, random_state=1)
    trainSet = []
    verifySet = []
    n = 1
    for train, test in kf.split(all_peptide):
        trainSet.append(pd2Dataset(all_peptide, train, ))
        verifySet.append(pd2Dataset(all_peptide, test))
        print(datetime.datetime.now().strftime(
            "%b %d %Y %H:%M:%S"), 'kf.split', n)
        n += 1
    return trainSet, verifySet, testset


def k_fold_split(all_peptide_file,  kf_num, slice_loc):
    all_peptide = pd.read_csv(all_peptide_file, header=0, names=['seq', 'interaction'])
    all_peptide = shuffle(all_peptide)
    
    if slice_loc[0] == 1:
        all_peptide = all_peptide[0:slice_loc[1]]
    print(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S"),
          "load dataset ", len(all_peptide), ' lenth')
    kf = KFold(kf_num, shuffle=True, random_state=1)
    trainSet = []
    verifySet = []
    n = 1
    for train, test in kf.split(all_peptide):
        trainSet.append(pd2Dataset(all_peptide, train, ))
        verifySet.append(pd2Dataset(all_peptide, test))
        print(datetime.datetime.now().strftime(
            "%b %d %Y %H:%M:%S"), 'kf.split', n)
        n += 1
    return trainSet, verifySet

def pd2Dataset(all_peptide, index):
    interaction = [all_peptide['interaction'].to_list()[i] for i in index]
    seq = [all_peptide['seq'].to_list()[i] for i in index]
    dataset_all = AMPDataset(seq, interaction)
    return dataset_all
