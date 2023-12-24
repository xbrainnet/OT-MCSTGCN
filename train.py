import random
import numpy as np
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader
import warnings

from data_process import ADNI
from model import STGCN_model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



k = 3
num_of_timesteps = 5
num_of_chev_filters = 45
num_of_time_filters = 10
time_conv_strides = 1
time_conv_kernel = 3
num_of_vertices = 90
num_of_features = 90
nhid1 = 1024
nhid2 = 256
n_class = 2
avg_acc = 0
avg_spe = 0
avg_recall = 0
avg_f1 = 0
avg_auc = 0
pre_ten = []
label_ten = []
gailv_ten = []

kk = 10

def stest(model, datasets_test):
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    gailv_all = []
    pro_all = []
    model.eval()  
    for net, data_feas, label in datasets_test:
        net, data_feas, label = net.to(DEVICE), data_feas.to(DEVICE), label.to(DEVICE)
        net = net.float()
        data_feas = data_feas.float()

        label = label.long()
        outs = model(net, data_feas)

        losss = F.nll_loss(outs, label)
        eval_loss += float(losss)
        gailv, pred = outs.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / net.shape[0]
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(outs[:, 1].cpu().detach().numpy())
    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all


i = 0
test_acc = []
test_pre = []
test_recall = []
test_f1 = []
test_auc = []
label_ten = []
test_sens = []
test_spec = []
pro_ten = []
dataset = ADNI()
train_ratio = 0.8
valid_ratio = 0.2
KF = KFold(n_splits=10, shuffle=True)
for train_idx, test_idx in KF.split(dataset):
    train_size = int(train_ratio * len(train_idx))
    valid_size = len(train_idx) - train_size
    train_indices, valid_indices = train_idx[:train_size], train_idx[train_size:]
    datasets_train = DataLoader(dataset, batch_size=20, shuffle=False, sampler=SubsetRandomSampler(train_indices))
    datasets_valid = DataLoader(dataset, batch_size=20, shuffle=False, sampler=SubsetRandomSampler(valid_indices))
    datasets_test = DataLoader(dataset, batch_size=20, shuffle=False, sampler=SubsetRandomSampler(test_idx))
    epoch = 300
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    patiences = 30
    min_acc = 0

    model = STGCN_model(k, num_of_timesteps, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                        time_conv_kernel, num_of_vertices, num_of_features, nhid1, nhid2, n_class)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)  # 0.005
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    for e in range(epoch):
        train_loss = 0
        train_acc = 0
        model.train()
        for ot_net, cheb, label in datasets_train:
            ot_net, cheb, label = ot_net.to(DEVICE), cheb.to(DEVICE), label.to(DEVICE)


            ot_net = ot_net.float()
            cheb = cheb.float()
            label = label.long()

            out = model(ot_net, cheb)  # torch.Size([4, 3])
            loss = F.nll_loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
            _, pred = out.max(1)
            num_correct = (pred == label).sum()
            acc = num_correct / ot_net.shape[0]
            train_acc += acc
        # scheduler.step()

        losses.append(train_loss / len(datasets_train))
        acces.append(train_acc / len(datasets_train))

        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest(
            model, datasets_valid)
        if eval_acc_epoch > min_acc:
            min_acc = eval_acc_epoch
            torch.save(model.state_dict(), './latest' + str(i) + '.pth')
            print("Model saved at epoch{}".format(e))

            patience = 0
        else:
            patience += 1
        if patience > patiences:
            break
        eval_losses.append(eval_loss / len(datasets_test))
        eval_acces.append(eval_acc / len(datasets_test))
        #     print('Eval Loss: {:.6f}, Eval Acc: {:.6f}'
        #           .format(eval_loss / len(datasets_test), eval_acc / len(datasets_test)))
        # '''
        print(
            'i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
            ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
            .format(i, e, train_loss / len(datasets_train), train_acc / len(datasets_train),
                    eval_loss / len(datasets_valid), eval_acc_epoch, precision, recall, f1, my_auc))
    model_test = STGCN_model(k, num_of_timesteps, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                             time_conv_kernel, num_of_vertices, num_of_features, nhid1, nhid2, n_class)
    model_test = model_test.to(DEVICE)
    model_test.load_state_dict(torch.load('./latest' + str(i) + '.pth'))  # 84.3750
    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest(
        model_test, datasets_test)

    test_acc.append(eval_acc_epoch)
    test_pre.append(precision)
    test_recall.append(recall)
    test_f1.append(f1)
    test_auc.append(my_auc)
    test_sens.append(sensitivity)
    test_spec.append(specificity)
    label_ten.extend(labels_all)
    pro_ten.extend(pro_all)

    i = i + 1
print("test_acc", test_acc)
print("test_pre", test_pre)
print("test_recall", test_recall)
print("test_f1", test_f1)
print("test_auc", test_auc)
print("test_sens", test_sens)
print("test_spec", test_spec)
avg_acc = sum(test_acc) / kk
avg_pre = sum(test_pre) / kk
avg_recall = sum(test_recall) / kk
avg_f1 = sum(test_f1) / kk
avg_auc = sum(test_auc) / kk
avg_sens = sum(test_sens) / kk
avg_spec = sum(test_spec) / kk
print("*****************************************************")
print('acc', avg_acc)
print('pre', avg_pre)
print('recall', avg_recall)
print('f1', avg_f1)
print('auc', avg_auc)
print("sensitivity", avg_sens)
print("specificity", avg_spec)

acc_std = np.sqrt(np.var(test_acc))
pre_std = np.sqrt(np.var(test_pre))
recall_std = np.sqrt(np.var(test_recall))
f1_std = np.sqrt(np.var(test_f1))
auc_std = np.sqrt(np.var(test_auc))
sens_std = np.sqrt(np.var(test_sens))
spec_std = np.sqrt(np.var(test_spec))
print("*****************************************************")
print("acc_std", acc_std)
print("pre_std", pre_std)
print("recall_std", recall_std)
print("f1_std", f1_std)
print("auc_std", auc_std)
print("sens_std", sens_std)
print("spec_std", spec_std)
print("*****************************************************")

print(label_ten)
print(pro_ten)
