from dataset import DrugSynergyDataset
from model import CATSynModel
import numpy as np
import torch
import pandas as pd
import dgl
import os
import time
from datetime import datetime
import logging
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score

def calc_stat(numbers):
    mu = sum(numbers) / len(numbers)
    sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
    return mu, sigma


def conf_inv(mu, sigma, n):
    delta = 2.776 * sigma / (n ** 0.5)  # 95%
    return mu - delta, mu + delta

data = DrugSynergyDataset('loewe')

test_res = []
test_label = []
test_losses = []
test_pccs = []
n_delimiter = 60
class_stats = np.zeros((10, 7))

log_file = 'cv.log'
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=logging.INFO)

for test_fold in range(10):
    tgl, vgl, egl, tg, vg, eg, tm, vm, em= data.get_graph(test_fold, 'random')
    drug_feat, cell_feat, drugslist, drugscount, cellscount = data.get_feat()
    cell_feat = torch.Tensor(cell_feat)
    model = CATSynModel(drug_feat.shape[1], cell_feat.shape[1], 512, 4, 512)
    device = torch.device('cuda:{:d}'.format(0))
    torch.cuda.set_device(device)
    model.to(device)
    
    tg = tg.to(device)
    vg = vg.to(device)
    eg = eg.to(device)
    tm = tm.to(device)
    vm = vm.to(device)
    em = em.to(device)

    for i in range(cellscount):
        tgl[i] = tgl[i].to(device)
        vgl[i] = vgl[i].to(device)
        egl[i] = egl[i].to(device)

    drug_feat = drug_feat.to(device)
    cell_feat = cell_feat.to(device)
    lbd = 1
    opt = torch.optim.Adam([{'params':model.parameters()}], lr=1e-4)
    min_loss = 1e9
    min_epoch = 0
    loss_func = nn.MSELoss(reduction='sum')

    for epoch in range(10000):
        loss = 0
        model.train()
        spec_h = []
        h = model.inference(tg, drug_feat, cell_feat).unsqueeze(-1)
        h = h.reshape(cellscount * drugscount + cellscount + drugscount, -1)

        hc = h[:cellscount * drugscount].reshape(cellscount, drugscount, -1)
        ncf = h[cellscount * drugscount:cellscount * drugscount + cellscount].reshape(cellscount, -1)
        hud = h[cellscount * drugscount + cellscount:].reshape(drugscount, -1)
        tot_l = 0
        
        lvx = []
        lvy = []
        mse_l = 0
        pcc_l = 0
        edge_type = ['syn','add','ant']
        
        for cellidx in range(cellscount):
            dec_graph = tgl[cellidx]
            logits = model.predict(dec_graph, hud, cell_feat[cellidx], hc[cellidx], drug_feat, ncf[cellidx])
            l_list = []
            l_list.append(logits[('drug','syn','drug')])
            l_list.append(logits[('drug','add','drug')])
            l_list.append(logits[('drug','ant','drug')])
            logits = torch.cat(l_list)

            t_list = []
            t_list.append(dec_graph.edata['syn'][('drug','syn','drug')])
            t_list.append(dec_graph.edata['syn'][('drug','add','drug')])
            t_list.append(dec_graph.edata['syn'][('drug','ant','drug')])
            edge_label = torch.cat(t_list)
            loss += loss_func(logits, edge_label)
            vx = logits - torch.mean(logits)
            vy = edge_label - torch.mean(edge_label)
            if (torch.sum(vy ** 2) > 1e-7):
                loss += logits.shape[0] * 8000 * (1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
                mse_l += logits.shape[0] * 8000 * (1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
                tot_l += logits.shape[0]

        pcc_l = loss.item() - mse_l
        pcc_l = 1.0 * pcc_l / tot_l
        mse_l = 1.0 * mse_l / tot_l

        opt.zero_grad()
        loss.backward()
        opt.step()

        print('Epoch : {}, train loss : {}'.format(epoch, loss.item() / tot_l))
        print('Epoch : {}, train mse loss : {}, pcc loss : {}'.format(epoch, pcc_l, mse_l))
        with torch.no_grad():
            model.eval()
            val_loss = 0
            mse = 0
            pcc = 0
            h = model.inference(tg, drug_feat, cell_feat).unsqueeze(-1)
            h = h.reshape(cellscount * drugscount + cellscount + drugscount, -1).detach()
            ncf = h[cellscount * drugscount:cellscount * drugscount + cellscount].reshape(cellscount, -1)
            hc = h[:cellscount * drugscount].reshape(cellscount, drugscount, -1)
            hud = h[cellscount * drugscount + cellscount:]
            tot_l = 0

            for cellidx in range(cellscount):
                dec_graph = vgl[cellidx]
                logits = model.predict(dec_graph, hud, cell_feat[cellidx], hc[cellidx], drug_feat, ncf[cellidx])
                l_list = []
                l_list.append(logits[('drug','syn','drug')])
                l_list.append(logits[('drug','add','drug')])
                l_list.append(logits[('drug','ant','drug')])
                logits = torch.cat(l_list).detach()

                t_list = []
                t_list.append(dec_graph.edata['syn'][('drug','syn','drug')])
                t_list.append(dec_graph.edata['syn'][('drug','add','drug')])
                t_list.append(dec_graph.edata['syn'][('drug','ant','drug')])
                edge_label = torch.cat(t_list)
                val_loss += loss_func(logits, edge_label)
                mse += loss_func(logits, edge_label)
                vx = logits - torch.mean(logits)
                vy = edge_label - torch.mean(edge_label)
                if (torch.sum(vy ** 2) > 1e-7):
                    val_loss += logits.shape[0] * 200 * (1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
                    pcc += logits.shape[0] * torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    tot_l += logits.shape[0]
                else:
                    val_loss += logits.shape[0] * 200 * 1
                    pcc += 0
                    tot_l += logits.shape[0]
            
            val_loss = val_loss.item()
            print('Epoch : {}, valid loss : {}'.format(epoch, mse / tot_l))
            print('Pcc : {}'.format(pcc / tot_l))
            
            if (val_loss < min_loss):
                min_loss = val_loss
                min_epoch = epoch
                print('Current best loss : {}, Epoch : {}'.format(mse / tot_l, min_epoch))
                torch.save(model.state_dict(), '../ckpt/best_infer_model_{}.ckpt'.format(test_fold))
                    
            if epoch > min_epoch + 2000:
                break
            

    model.load_state_dict(torch.load('../ckpt/best_infer_model_{}.ckpt'.format(test_fold)))
    model.eval()
    
    with torch.no_grad():
        spec_h = []
        h = model.inference(tg, drug_feat, cell_feat).unsqueeze(-1)
        h = h.reshape(cellscount * drugscount + cellscount + drugscount, -1).detach()
        ncf = h[cellscount * drugscount:cellscount * drugscount + cellscount].reshape(cellscount, -1)
        hc = h[:cellscount * drugscount].reshape(cellscount, drugscount, -1)
        hud = h[cellscount * drugscount + cellscount:]
        y_true = []
        y_pred = []

        tot_l = 0

        for cellidx in range(cellscount):
            dec_graph = egl[cellidx]
            logits = model.predict(dec_graph, hud, cell_feat[cellidx], hc[cellidx], drug_feat, ncf[cellidx])
            l_list = []
            l_list.append(logits[('drug','syn','drug')])
            l_list.append(logits[('drug','add','drug')])
            l_list.append(logits[('drug','ant','drug')])
            logits = torch.cat(l_list).detach()

            t_list = []
            t_list.append(dec_graph.edata['syn'][('drug','syn','drug')])
            t_list.append(dec_graph.edata['syn'][('drug','add','drug')])
            t_list.append(dec_graph.edata['syn'][('drug','ant','drug')])
            edge_label = torch.cat(t_list)
            y_pred.append(logits)
            
            y_true.append(edge_label)

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        test_loss = loss_func(y_pred, y_true).item()
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().flatten()
        test_pcc = np.corrcoef(y_pred, y_true)[0, 1]
        test_loss /= len(y_true)
        y_pred_binary = [ 1 if x >= 2.64 else 0 for x in y_pred ]
        y_true_binary = [ 1 if x >= 2.64 else 0 for x in y_true ]
        roc_score = 0
        try:
            roc_score = roc_auc_score(y_true_binary, y_pred)
        except ValueError:
            pass
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        auprc_score = auc(recall, precision)
        accuracy = accuracy_score( y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary)
        kappa = cohen_kappa_score(y_true_binary, y_pred_binary)

    class_stat = [roc_score, auprc_score, accuracy, f1, precision, recall, kappa]
    class_stats[test_fold] = class_stat
    test_losses.append(test_loss)
    test_pccs.append(test_pcc)
    logging.info("Test loss: {:.4f}".format(test_loss))
    logging.info("Test pcc: {:.4f}".format(test_pcc))
    logging.info("*" * n_delimiter + '\n')
    
     
logging.info("CV completed")
mu, sigma = calc_stat(test_losses)
logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
lo, hi = conf_inv(mu, sigma, len(test_losses))
logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
rmse_loss = [x ** 0.5 for x in test_losses]
mu, sigma = calc_stat(rmse_loss)
logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
pcc_mean, pcc_std = calc_stat(test_pccs)
logging.info("pcc: {:.4f} ± {:.4f}".format(pcc_mean, pcc_std))

class_stats = np.concatenate([class_stats, class_stats.mean(axis=0, keepdims=True), class_stats.std(axis=0, keepdims=True)], axis=0)
pd.DataFrame(class_stats).to_csv('class_stats.txt', sep='\t', header=None, index=None)

