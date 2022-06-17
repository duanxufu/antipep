import os

import torch


def save_model(root, model, name, k, title,p:list):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, name)):
        os.mkdir(os.path.join(root, name))
    if not os.path.exists(os.path.join(root, name, title)):
        os.mkdir(os.path.join(root, name, title))
    if not os.path.exists(os.path.join(root, name, title, 'kf_'+str(k))):
        os.mkdir(os.path.join(root, name, title, 'kf_'+str(k)))
    torch.save(model, os.path.join(root, name, title, 'kf_'+str(k), 'model.pt'))
    print(os.path.join(root, name, title, 'kf_'+str(k), 'model.pt'))
    lt =['loss', 'acc', 'precision', 'recall', 'f1', 'mcc', 'roc', 'auc']
    f = open(os.path.join(root, name, title, 'kf_'+str(k), 'param.txt'),'a+')
    for i in range(len(p)):
        f.write(str(lt[i])+": ")
        f.write(str(p[i])+": ")
    f.write('\n')
    f.close()

def load_model(root, name, k, title):
    model = torch.load(os.path.join(
        root, name, title, 'kf_'+str(k), 'model.pt'))
    return model


def save_param(root, name, title, ave):

    f = open(os.path.join(root, name, title, 'best_param.txt'), 'a')
    f.write('test loss:')
    f.write(str(ave[0]))
    f.write('test acc : ')
    f.write(str(ave[1]))
    f.write('test_recall: ')
    f.write(str(ave[2]))
    f.write('test_f1: ')
    f.write(str(ave[3]))
    f.write('test_precision: ')
    f.write(str(ave[4]))
    f.write('test mcc: ')
    f.write(str(ave[5]))
    f.write('test roc: ')
    f.write(str(ave[6]))
    f.write('test auc: ')
    f.write(str(ave[7]))
    f.close()
    print('-----------------------------------------------\n---test---')
    print('test loss:', ave[0]/ave[8], 'test acc : ', ave[1]/ave[8], 'recall: ', ave[2]/ave[8],
          'f1: ', ave[3]/ave[8], 'precision', ave[4]/ave[8], 'test mcc: ', ave[5]/ave[8],
           'test roc:', ave[6]/ave[8], 'test auc: ', ave[7]/ave[8])
    return
