import time
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import dgl

class DrugSynergyDataset():
    def __init__(self, dataset='loewe'):
        if dataset == 'loewe':
            self.data = pd.read_csv('../rawdata/oneil_loewe_cutoff30_2.txt', sep='\t')
        elif dataset == 'bliss':
            self.data = pd.read_csv('../rawdata/oneil_synergy_bliss_2.txt', sep='\t')
        elif dataset == 'hsa':
            self.data = pd.read_csv('../rawdata/oneil_synergy_hsa_2.txt', sep='\t')
        elif dataset == 'zip':
            self.data = pd.read_csv('../rawdata/oneil_synergy_zip_2.txt', sep='\t')
        self.dataset = dataset
        self.drugslist = sorted(list(set(list(self.data['drugname1']) + list(self.data['drugname2'])))) #38
        self.drugscount = len(self.drugslist)
        self.cellslist = sorted(list(set(self.data['cell_line']))) 
        self.cellscount = len(self.cellslist)

        self.drug_feat = pd.read_csv('../rawdata/oneil_drug_informax_feat.txt',sep='\t', header=None)
        self.drug_feat = torch.Tensor(np.array(self.drug_feat))
        self.cell_feat = np.load('../rawdata/oneil_cell_feat.npy')

    def get_feat(self):
        return self.drug_feat, self.cell_feat, self.drugslist, self.drugscount, self.cellscount

    def get_graph(self, test_fold, cv_type):
        valid_fold = list(range(10))[test_fold-1]
        train_fold = [ x for x in list(range(10)) if x != test_fold and x != valid_fold ]

        train_g_list = []
        valid_g_list = []
        test_g_list = []
        if (cv_type == 'random'):            
            if self.dataset == 'loewe':
                upb = 30
                lowb = 0
            elif self.dataset == 'bliss':
                upb = 3.68
                lowb = -3.37
            elif self.dataset == 'hsa':
                upb = 3.87
                lowb = -3.02
            elif self.dataset == 'zip':
                upb = 2.64
                lowb = -4.48
            train_g_list = []
            
            for cellidx in range(self.cellscount):
                # cellidx = 0
                cellname = self.cellslist[cellidx]
                print('processing ', cellname)
                each_data = self.data[self.data['cell_line']==cellname]
                edges_src = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
                edges_dst = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
                edge_val = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

                for each in each_data.values:
                    drugname1, drugname2, cell_line, synergy, fold, d1_fold, d2_fold, c_fold = each
                    drugidx1 = self.drugslist.index(drugname1)
                    drugidx2 = self.drugslist.index(drugname2)

                    if float(synergy) >= upb: #syn
                        if fold in train_fold:
                            edges_src[0][0].append(drugidx1)
                            edges_dst[0][0].append(drugidx2)
                            edges_src[0][0].append(drugidx2)
                            edges_dst[0][0].append(drugidx1)
                            edge_val[0][0].append(synergy)
                            edge_val[0][0].append(synergy)
                        elif fold == valid_fold:
                            edges_src[1][0].append(drugidx1)
                            edges_dst[1][0].append(drugidx2)
                            edges_src[1][0].append(drugidx2)
                            edges_dst[1][0].append(drugidx1)
                            edge_val[1][0].append(synergy)
                            edge_val[1][0].append(synergy)
                        elif fold == test_fold:
                            edges_src[2][0].append(drugidx1)
                            edges_dst[2][0].append(drugidx2)
                            edges_src[2][0].append(drugidx2)
                            edges_dst[2][0].append(drugidx1)
                            edge_val[2][0].append(synergy)
                            edge_val[2][0].append(synergy)
                    elif (float(synergy) < upb) and (float(synergy) > lowb): #add
                        if fold in train_fold:
                            edges_src[0][1].append(drugidx1)
                            edges_dst[0][1].append(drugidx2)
                            edges_src[0][1].append(drugidx2)
                            edges_dst[0][1].append(drugidx1)
                            edge_val[0][1].append(synergy)
                            edge_val[0][1].append(synergy)
                        elif fold == valid_fold:
                            edges_src[1][1].append(drugidx1)
                            edges_dst[1][1].append(drugidx2)
                            edges_src[1][1].append(drugidx2)
                            edges_dst[1][1].append(drugidx1)
                            edge_val[1][1].append(synergy)
                            edge_val[1][1].append(synergy)
                        elif fold == test_fold:
                            edges_src[2][1].append(drugidx1)
                            edges_dst[2][1].append(drugidx2)
                            edges_src[2][1].append(drugidx2)
                            edges_dst[2][1].append(drugidx1)
                            edge_val[2][1].append(synergy)
                            edge_val[2][1].append(synergy)
                    elif float(synergy) < lowb:#ant
                        if fold in train_fold:
                            edges_src[0][2].append(drugidx1)
                            edges_dst[0][2].append(drugidx2)
                            edges_src[0][2].append(drugidx2)
                            edges_dst[0][2].append(drugidx1)
                            edge_val[0][2].append(synergy)
                            edge_val[0][2].append(synergy)
                        elif fold == valid_fold:
                            edges_src[1][2].append(drugidx1)
                            edges_dst[1][2].append(drugidx2)
                            edges_src[1][2].append(drugidx2)
                            edges_dst[1][2].append(drugidx1)
                            edge_val[1][2].append(synergy)
                            edge_val[1][2].append(synergy)
                        elif fold == test_fold:
                            edges_src[2][2].append(drugidx1)
                            edges_dst[2][2].append(drugidx2)
                            edges_src[2][2].append(drugidx2)
                            edges_dst[2][2].append(drugidx1)
                            edge_val[2][2].append(synergy)
                            edge_val[2][2].append(synergy)

                for i in range(self.drugscount):
                    for j in range(3):
                        edges_src[j][3].append(i)
                        edges_dst[j][3].append(i)
                        edge_val[j][3].append(0)
                

                train_graph_data = {}
                src = torch.LongTensor(edges_src[0][0])
                dst = torch.LongTensor(edges_dst[0][0])
                train_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][1])
                dst = torch.LongTensor(edges_dst[0][1])
                train_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][2])
                dst = torch.LongTensor(edges_dst[0][2])
                train_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][3])
                dst = torch.LongTensor(edges_dst[0][3])
                train_graph_data[('drug','intra','drug')] = (src, dst)

                train_graph = dgl.heterograph(train_graph_data)
                train_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[0][0]),('drug','add','drug'):torch.Tensor(edge_val[0][1]),('drug','ant','drug'):torch.Tensor(edge_val[0][2]),('drug','intra','drug'):torch.Tensor(edge_val[0][3])}

                train_g_list.append(train_graph)

                valid_graph_data = {}
                src = torch.LongTensor(edges_src[1][0])
                dst = torch.LongTensor(edges_dst[1][0])
                valid_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][1])
                dst = torch.LongTensor(edges_dst[1][1])
                valid_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][2])
                dst = torch.LongTensor(edges_dst[1][2])
                valid_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][3])
                dst = torch.LongTensor(edges_dst[1][3])
                valid_graph_data[('drug','intra','drug')] = (src, dst)

                valid_graph = dgl.heterograph(valid_graph_data)

                valid_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[1][0]),('drug','add','drug'):torch.Tensor(edge_val[1][1]),('drug','ant','drug'):torch.Tensor(edge_val[1][2]),('drug','intra','drug'):torch.Tensor(edge_val[1][3])}
                valid_g_list.append(valid_graph)

                test_graph_data = {}
                src = torch.LongTensor(edges_src[2][0])
                dst = torch.LongTensor(edges_dst[2][0])
                test_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][1])
                dst = torch.LongTensor(edges_dst[2][1])
                test_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][2])
                dst = torch.LongTensor(edges_dst[2][2])
                test_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][3])
                dst = torch.LongTensor(edges_dst[2][3])
                test_graph_data[('drug','intra','drug')] = (src, dst)

                test_graph = dgl.heterograph(test_graph_data)
                test_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[2][0]),('drug','add','drug'):torch.Tensor(edge_val[2][1]),('drug','ant','drug'):torch.Tensor(edge_val[2][2]),('drug','intra','drug'):torch.Tensor(edge_val[2][3])}
                test_g_list.append(test_graph)
        

            d = np.zeros([self.cellscount, self.drugscount, self.drugscount])
            edges_src = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            edges_dst = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            edge_val = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

            for each in self.data.values:
                drugname1, drugname2, cell_line, synergy, fold, d1_fold, d2_fold, c_fold = each
                cellidx = self.cellslist.index(cell_line)
                drugidx1 = self.drugslist.index(drugname1)
                drugidx2 = self.drugslist.index(drugname2)
                if fold in train_fold:
                    if float(synergy) >= upb:
                        d[cellidx][drugidx1][drugidx2] = 1
                        d[cellidx][drugidx2][drugidx1] = 1
                    elif float(synergy) >= 0:
                        d[cellidx][drugidx1][drugidx2] = 2
                        d[cellidx][drugidx2][drugidx1] = 2
                    else:
                        d[cellidx][drugidx1][drugidx2] = 3
                        d[cellidx][drugidx2][drugidx1] = 3
                if float(synergy) >= upb: #syn
                    if fold in train_fold:
                        edges_src[0][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][0].append(synergy)
                        edge_val[0][0].append(synergy)

                    elif fold == valid_fold:
                        edges_src[1][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][0].append(synergy)
                        edge_val[1][0].append(synergy)
                    elif fold == test_fold:
                        edges_src[2][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][0].append(synergy)
                        edge_val[2][0].append(synergy)
                elif (float(synergy) > 0): #add
                    if fold in train_fold:
                        edges_src[0][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][1].append(synergy)
                        edge_val[0][1].append(synergy)
                    elif fold == valid_fold:
                        edges_src[1][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][1].append(synergy)
                        edge_val[1][1].append(synergy)
                    elif fold == test_fold:
                        edges_src[2][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][1].append(synergy)
                        edge_val[2][1].append(synergy)
                else:#ant
                    if fold in train_fold:
                        edges_src[0][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][2].append(synergy)
                        edge_val[0][2].append(synergy)
                    elif fold == valid_fold:
                        edges_src[1][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][2].append(synergy)
                        edge_val[1][2].append(synergy)
                    elif fold == test_fold:
                        edges_src[2][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][2].append(synergy)
                        edge_val[2][2].append(synergy)


            for cid in range(self.cellscount * self.drugscount + self.cellscount + self.drugscount):
                for i in range(3):
                    edges_src[i][3].append(cid)
                    edges_dst[i][3].append(cid)
                    edge_val[i][3].append(0)

            '''
            universal: cell c*d + i
            drug  c*d+c + i
            '''

            
            for cid in range(self.cellscount):
                for did in range(self.drugscount):
                    edges_dst[0][3].append(self.cellscount * self.drugscount + cid)
                    edges_src[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    edges_src[0][3].append(self.cellscount * self.drugscount + cid)
                    edges_dst[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    
                    edges_dst[0][3].append(self.cellscount * self.drugscount + self.cellscount + did)
                    edges_src[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    edges_src[0][3].append(self.cellscount * self.drugscount + self.cellscount + did)
                    edges_dst[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    

            for did in range(self.drugscount):
                for did2 in range(self.drugscount):
                    if did != did2:
                        edges_dst[0][3].append(self.cellscount * self.drugscount + cid + did)
                        edges_src[0][3].append(self.cellscount * self.drugscount + cid + did2)
                        edge_val[0][3].append(0)

            

            for i in range(self.cellscount):
                print(i)
                for j in range(self.cellscount):
                    for k in range(self.drugscount):
                        for l in range(k, self.drugscount):
                            if (i==j) and (k==l):
                                continue
                            union = 0
                            its = 0
                            for x in range(self.drugscount):
                                if (x == k) or (x == l):
                                    continue
                                if (d[i][k][x] == d[j][l][x]) and (d[i][k][x] != 0):
                                    union += 1
                                    its += 1
                                else:
                                    if (d[i][k][x] != 0):
                                        union += 1
                                    if (d[j][l][x] != 0):
                                        union += 1
                            if union == 0:
                                scr = 0.0
                            else:
                                scr = 1.0 * its / union
                            if scr >= 0.6:
                                edges_src[0][3].append(i * self.drugscount + k)
                                edges_dst[0][3].append(j * self.drugscount + l)
                                edges_src[0][3].append(j * self.drugscount + l)
                                edges_dst[0][3].append(i * self.drugscount + k)
                                edge_val[0][3].append(0)
                                edge_val[0][3].append(0)

            train_mask = []
            valid_mask = []
            test_mask = []

            train_graph_data = {}
            src = torch.LongTensor(edges_src[0][0])
            dst = torch.LongTensor(edges_dst[0][0])
            train_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[0][1])
            dst = torch.LongTensor(edges_dst[0][1])
            train_graph_data[('drug','add','drug')] = (src, dst)
            train_mask.append(torch.ones([len(edges_src[0][1])], dtype=torch.bool))
            #train_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[0][2])
            dst = torch.LongTensor(edges_dst[0][2])
            train_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[0][3])
            dst = torch.LongTensor(edges_dst[0][3])
            train_graph_data[('drug','intra','drug')] = (src, dst)
            
            train_graph = dgl.heterograph(train_graph_data)
            train_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[0][0]),('drug','add','drug'):torch.Tensor(edge_val[0][1]),('drug','ant','drug'):torch.Tensor(edge_val[0][2]),('drug','intra','drug'):torch.Tensor(edge_val[0][3])}

            #train_g_list.append(train_graph)

            valid_graph_data = {}
            src = torch.LongTensor(edges_src[1][0])
            dst = torch.LongTensor(edges_dst[1][0])
            valid_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[1][1])
            dst = torch.LongTensor(edges_dst[1][1])
            valid_graph_data[('drug','add','drug')] = (src, dst)
            valid_mask.append(torch.ones([len(edges_src[1][1])], dtype=torch.bool))
            #valid_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[1][2])
            dst = torch.LongTensor(edges_dst[1][2])
            valid_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[1][3])
            dst = torch.LongTensor(edges_dst[1][3])
            valid_graph_data[('drug','intra','drug')] = (src, dst)
            valid_graph = dgl.heterograph(valid_graph_data)

            valid_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[1][0]),('drug','add','drug'):torch.Tensor(edge_val[1][1]),('drug','ant','drug'):torch.Tensor(edge_val[1][2]),('drug','intra','drug'):torch.Tensor(edge_val[1][3])}
            #valid_g_list.append(valid_graph)

            test_graph_data = {}
            src = torch.LongTensor(edges_src[2][0])
            dst = torch.LongTensor(edges_dst[2][0])
            test_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[2][1])
            dst = torch.LongTensor(edges_dst[2][1])
            test_graph_data[('drug','add','drug')] = (src, dst)
            test_mask.append(torch.ones([len(edges_src[2][1])], dtype=torch.bool))
            #test_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[2][2])
            dst = torch.LongTensor(edges_dst[2][2])
            test_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[2][3])
            dst = torch.LongTensor(edges_dst[2][3])
            test_graph_data[('drug','intra','drug')] = (src, dst)
            test_graph = dgl.heterograph(test_graph_data)
            test_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[2][0]),('drug','add','drug'):torch.Tensor(edge_val[2][1]),('drug','ant','drug'):torch.Tensor(edge_val[2][2]),('drug','intra','drug'):torch.Tensor(edge_val[2][3])}
            #test_g_list.append(test_graph)
            train_mask = torch.cat(train_mask)
            valid_mask = torch.cat(valid_mask)
            test_mask = torch.cat(test_mask)
            return train_g_list, valid_g_list, test_g_list, train_graph, valid_graph, test_graph,train_mask, valid_mask, test_mask#, train_g_list
        elif (cv_type == 'cellline'):
            if self.dataset == 'loewe':
                upb = 30
                lowb = 0
            elif self.dataset == 'bliss':
                upb = 3.68
                lowb = -3.37
            elif self.dataset == 'hsa':
                upb = 3.87
                lowb = -3.02
            elif self.dataset == 'zip':
                upb = 2.64
                lowb = -4.48
            train_g_list = []
            
            for cellidx in range(self.cellscount):
                # cellidx = 0
                cellname = self.cellslist[cellidx]
                print('processing ', cellname)
                each_data = self.data[self.data['cell_line']==cellname]
                edges_src = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
                edges_dst = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
                edge_val = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

                for each in each_data.values:
                    drugname1, drugname2, cell_line, synergy, fold, d1_fold, d2_fold, c_fold = each
                    drugidx1 = self.drugslist.index(drugname1)
                    drugidx2 = self.drugslist.index(drugname2)

                    if float(synergy) >= upb: #syn
                        if c_fold in train_fold:
                            edges_src[0][0].append(drugidx1)
                            edges_dst[0][0].append(drugidx2)
                            edges_src[0][0].append(drugidx2)
                            edges_dst[0][0].append(drugidx1)
                            edge_val[0][0].append(synergy)
                            edge_val[0][0].append(synergy)
                        elif c_fold == valid_fold:
                            edges_src[1][0].append(drugidx1)
                            edges_dst[1][0].append(drugidx2)
                            edges_src[1][0].append(drugidx2)
                            edges_dst[1][0].append(drugidx1)
                            edge_val[1][0].append(synergy)
                            edge_val[1][0].append(synergy)
                        elif c_fold == test_fold:
                            edges_src[2][0].append(drugidx1)
                            edges_dst[2][0].append(drugidx2)
                            edges_src[2][0].append(drugidx2)
                            edges_dst[2][0].append(drugidx1)
                            edge_val[2][0].append(synergy)
                            edge_val[2][0].append(synergy)
                    elif (float(synergy) < upb) and (float(synergy) > lowb): #add
                        if c_fold in train_fold:
                            edges_src[0][1].append(drugidx1)
                            edges_dst[0][1].append(drugidx2)
                            edges_src[0][1].append(drugidx2)
                            edges_dst[0][1].append(drugidx1)
                            edge_val[0][1].append(synergy)
                            edge_val[0][1].append(synergy)
                        elif c_fold == valid_fold:
                            edges_src[1][1].append(drugidx1)
                            edges_dst[1][1].append(drugidx2)
                            edges_src[1][1].append(drugidx2)
                            edges_dst[1][1].append(drugidx1)
                            edge_val[1][1].append(synergy)
                            edge_val[1][1].append(synergy)
                        elif c_fold == test_fold:
                            edges_src[2][1].append(drugidx1)
                            edges_dst[2][1].append(drugidx2)
                            edges_src[2][1].append(drugidx2)
                            edges_dst[2][1].append(drugidx1)
                            edge_val[2][1].append(synergy)
                            edge_val[2][1].append(synergy)
                    elif float(synergy) < lowb:#ant
                        if c_fold in train_fold:
                            edges_src[0][2].append(drugidx1)
                            edges_dst[0][2].append(drugidx2)
                            edges_src[0][2].append(drugidx2)
                            edges_dst[0][2].append(drugidx1)
                            edge_val[0][2].append(synergy)
                            edge_val[0][2].append(synergy)
                        elif c_fold == valid_fold:
                            edges_src[1][2].append(drugidx1)
                            edges_dst[1][2].append(drugidx2)
                            edges_src[1][2].append(drugidx2)
                            edges_dst[1][2].append(drugidx1)
                            edge_val[1][2].append(synergy)
                            edge_val[1][2].append(synergy)
                        elif c_fold == test_fold:
                            edges_src[2][2].append(drugidx1)
                            edges_dst[2][2].append(drugidx2)
                            edges_src[2][2].append(drugidx2)
                            edges_dst[2][2].append(drugidx1)
                            edge_val[2][2].append(synergy)
                            edge_val[2][2].append(synergy)

                for i in range(self.drugscount):
                    edges_src[0][3].append(i)
                    edges_dst[0][3].append(i)
                    edge_val[0][3].append(0)
                    for j in range(1,3):
                        edges_src[j][3].append(i)
                        edges_dst[j][3].append(i)
                        edge_val[j][3].append(0)
                

                train_graph_data = {}
                src = torch.LongTensor(edges_src[0][0])
                dst = torch.LongTensor(edges_dst[0][0])
                train_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][1])
                dst = torch.LongTensor(edges_dst[0][1])
                train_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][2])
                dst = torch.LongTensor(edges_dst[0][2])
                train_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][3])
                dst = torch.LongTensor(edges_dst[0][3])
                train_graph_data[('drug','intra','drug')] = (src, dst)

                train_graph = dgl.heterograph(train_graph_data)
                train_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[0][0]),('drug','add','drug'):torch.Tensor(edge_val[0][1]),('drug','ant','drug'):torch.Tensor(edge_val[0][2]),('drug','intra','drug'):torch.Tensor(edge_val[0][3])}

                train_g_list.append(train_graph)

                valid_graph_data = {}
                src = torch.LongTensor(edges_src[1][0])
                dst = torch.LongTensor(edges_dst[1][0])
                valid_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][1])
                dst = torch.LongTensor(edges_dst[1][1])
                valid_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][2])
                dst = torch.LongTensor(edges_dst[1][2])
                valid_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][3])
                dst = torch.LongTensor(edges_dst[1][3])
                valid_graph_data[('drug','intra','drug')] = (src, dst)

                valid_graph = dgl.heterograph(valid_graph_data)

                valid_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[1][0]),('drug','add','drug'):torch.Tensor(edge_val[1][1]),('drug','ant','drug'):torch.Tensor(edge_val[1][2]),('drug','intra','drug'):torch.Tensor(edge_val[1][3])}
                valid_g_list.append(valid_graph)

                test_graph_data = {}
                src = torch.LongTensor(edges_src[2][0])
                dst = torch.LongTensor(edges_dst[2][0])
                test_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][1])
                dst = torch.LongTensor(edges_dst[2][1])
                test_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][2])
                dst = torch.LongTensor(edges_dst[2][2])
                test_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][3])
                dst = torch.LongTensor(edges_dst[2][3])
                test_graph_data[('drug','intra','drug')] = (src, dst)

                test_graph = dgl.heterograph(test_graph_data)
                test_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[2][0]),('drug','add','drug'):torch.Tensor(edge_val[2][1]),('drug','ant','drug'):torch.Tensor(edge_val[2][2]),('drug','intra','drug'):torch.Tensor(edge_val[2][3])}
                test_g_list.append(test_graph)
        

            d = np.zeros([self.cellscount, self.drugscount, self.drugscount])
            edges_src = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            edges_dst = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            edge_val = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

            for each in self.data.values:
                drugname1, drugname2, cell_line, synergy, fold, d1_fold, d2_fold, c_fold = each
                cellidx = self.cellslist.index(cell_line)
                drugidx1 = self.drugslist.index(drugname1)
                drugidx2 = self.drugslist.index(drugname2)
                if c_fold in train_fold:
                    if float(synergy) >= upb:
                        d[cellidx][drugidx1][drugidx2] = 1
                        d[cellidx][drugidx2][drugidx1] = 1
                    elif float(synergy) >= 0:
                        d[cellidx][drugidx1][drugidx2] = 2
                        d[cellidx][drugidx2][drugidx1] = 2
                    else:
                        d[cellidx][drugidx1][drugidx2] = 3
                        d[cellidx][drugidx2][drugidx1] = 3
                if float(synergy) >= upb: #syn
                    if c_fold in train_fold:
                        edges_src[0][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][0].append(synergy)
                        edge_val[0][0].append(synergy)

                    elif c_fold == valid_fold:
                        edges_src[1][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][0].append(synergy)
                        edge_val[1][0].append(synergy)
                    elif c_fold == test_fold:
                        edges_src[2][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][0].append(synergy)
                        edge_val[2][0].append(synergy)
                elif (float(synergy) > 0): #add
                    if c_fold in train_fold:
                        edges_src[0][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][1].append(synergy)
                        edge_val[0][1].append(synergy)
                    elif c_fold == valid_fold:
                        edges_src[1][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][1].append(synergy)
                        edge_val[1][1].append(synergy)
                    elif c_fold == test_fold:
                        edges_src[2][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][1].append(synergy)
                        edge_val[2][1].append(synergy)
                else:#ant
                    if c_fold in train_fold:
                        edges_src[0][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][2].append(synergy)
                        edge_val[0][2].append(synergy)
                    elif c_fold == valid_fold:
                        edges_src[1][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][2].append(synergy)
                        edge_val[1][2].append(synergy)
                    elif c_fold == test_fold:
                        edges_src[2][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][2].append(synergy)
                        edge_val[2][2].append(synergy)


            for cid in range(self.cellscount * self.drugscount + self.cellscount + self.drugscount):
                for i in range(3):
                    edges_src[i][3].append(cid)
                    edges_dst[i][3].append(cid)
                    edge_val[i][3].append(0)

            '''
            universal: cell c*d + i
            drug  c*d+c + i
            '''

            
            for cid in range(self.cellscount):
                for did in range(self.drugscount):
                    edges_dst[0][3].append(self.cellscount * self.drugscount + cid)
                    edges_src[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    edges_src[0][3].append(self.cellscount * self.drugscount + cid)
                    edges_dst[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    
                    edges_dst[0][3].append(self.cellscount * self.drugscount + self.cellscount + did)
                    edges_src[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    edges_src[0][3].append(self.cellscount * self.drugscount + self.cellscount + did)
                    edges_dst[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    

            for did in range(self.drugscount):
                for did2 in range(self.drugscount):
                    if did != did2:
                        edges_dst[0][3].append(self.cellscount * self.drugscount + cid + did)
                        edges_src[0][3].append(self.cellscount * self.drugscount + cid + did2)
                        edge_val[0][3].append(0)

            

            for i in range(0):#self.cellscount):
                print(i)
                for j in range(self.cellscount):
                    for k in range(self.drugscount):
                        for l in range(k, self.drugscount):
                            if (i==j) and (k==l):
                                continue
                            union = 0
                            its = 0
                            for x in range(self.drugscount):
                                if (x == k) or (x == l):
                                    continue
                                if (d[i][k][x] == d[j][l][x]) and (d[i][k][x] != 0):
                                    union += 1
                                    its += 1
                                else:
                                    if (d[i][k][x] != 0):
                                        union += 1
                                    if (d[j][l][x] != 0):
                                        union += 1
                            if union == 0:
                                scr = 0.0
                            else:
                                scr = 1.0 * its / union
                            if scr >= 0.8:
                                edges_src[0][3].append(i * self.drugscount + k)
                                edges_dst[0][3].append(j * self.drugscount + l)
                                edges_src[0][3].append(j * self.drugscount + l)
                                edges_dst[0][3].append(i * self.drugscount + k)
                                edge_val[0][3].append(0)
                                edge_val[0][3].append(0)

            train_mask = []
            valid_mask = []
            test_mask = []

            train_graph_data = {}
            src = torch.LongTensor(edges_src[0][0])
            dst = torch.LongTensor(edges_dst[0][0])
            train_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[0][1])
            dst = torch.LongTensor(edges_dst[0][1])
            train_graph_data[('drug','add','drug')] = (src, dst)
            train_mask.append(torch.ones([len(edges_src[0][1])], dtype=torch.bool))
            #train_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[0][2])
            dst = torch.LongTensor(edges_dst[0][2])
            train_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[0][3])
            dst = torch.LongTensor(edges_dst[0][3])
            train_graph_data[('drug','intra','drug')] = (src, dst)
            
            train_graph = dgl.heterograph(train_graph_data)
            train_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[0][0]),('drug','add','drug'):torch.Tensor(edge_val[0][1]),('drug','ant','drug'):torch.Tensor(edge_val[0][2]),('drug','intra','drug'):torch.Tensor(edge_val[0][3])}

            #train_g_list.append(train_graph)

            valid_graph_data = {}
            src = torch.LongTensor(edges_src[1][0])
            dst = torch.LongTensor(edges_dst[1][0])
            valid_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[1][1])
            dst = torch.LongTensor(edges_dst[1][1])
            valid_graph_data[('drug','add','drug')] = (src, dst)
            valid_mask.append(torch.ones([len(edges_src[1][1])], dtype=torch.bool))
            #valid_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[1][2])
            dst = torch.LongTensor(edges_dst[1][2])
            valid_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[1][3])
            dst = torch.LongTensor(edges_dst[1][3])
            valid_graph_data[('drug','intra','drug')] = (src, dst)
            valid_graph = dgl.heterograph(valid_graph_data)

            valid_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[1][0]),('drug','add','drug'):torch.Tensor(edge_val[1][1]),('drug','ant','drug'):torch.Tensor(edge_val[1][2]),('drug','intra','drug'):torch.Tensor(edge_val[1][3])}
            #valid_g_list.append(valid_graph)

            test_graph_data = {}
            src = torch.LongTensor(edges_src[2][0])
            dst = torch.LongTensor(edges_dst[2][0])
            test_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[2][1])
            dst = torch.LongTensor(edges_dst[2][1])
            test_graph_data[('drug','add','drug')] = (src, dst)
            test_mask.append(torch.ones([len(edges_src[2][1])], dtype=torch.bool))
            #test_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[2][2])
            dst = torch.LongTensor(edges_dst[2][2])
            test_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[2][3])
            dst = torch.LongTensor(edges_dst[2][3])
            test_graph_data[('drug','intra','drug')] = (src, dst)
            test_graph = dgl.heterograph(test_graph_data)
            test_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[2][0]),('drug','add','drug'):torch.Tensor(edge_val[2][1]),('drug','ant','drug'):torch.Tensor(edge_val[2][2]),('drug','intra','drug'):torch.Tensor(edge_val[2][3])}
            #test_g_list.append(test_graph)
            train_mask = torch.cat(train_mask)
            valid_mask = torch.cat(valid_mask)
            test_mask = torch.cat(test_mask)
            return train_g_list, valid_g_list, test_g_list, train_graph, valid_graph, test_graph,train_mask, valid_mask, test_mask#, train_g_list
        elif (cv_type=='drug'):
            if self.dataset == 'loewe':
                upb = 30
                lowb = 0
            elif self.dataset == 'bliss':
                upb = 3.68
                lowb = -3.37
            elif self.dataset == 'hsa':
                upb = 3.87
                lowb = -3.02
            elif self.dataset == 'zip':
                upb = 2.64
                lowb = -4.48
            train_g_list = []
            
            for cellidx in range(self.cellscount):
                # cellidx = 0
                cellname = self.cellslist[cellidx]
                print('processing ', cellname)
                each_data = self.data[self.data['cell_line']==cellname]
                edges_src = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
                edges_dst = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
                edge_val = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

                for each in each_data.values:
                    drugname1, drugname2, cell_line, synergy, fold, d1_fold, d2_fold, c_fold = each
                    drugidx1 = self.drugslist.index(drugname1)
                    drugidx2 = self.drugslist.index(drugname2)

                    if float(synergy) >= upb: #syn
                        if (d1_fold in train_fold) and (d2_fold in train_fold):
                            edges_src[0][0].append(drugidx1)
                            edges_dst[0][0].append(drugidx2)
                            edges_src[0][0].append(drugidx2)
                            edges_dst[0][0].append(drugidx1)
                            edge_val[0][0].append(synergy)
                            edge_val[0][0].append(synergy)
                        elif (d1_fold == valid_fold) or (d2_fold == valid_fold):
                            edges_src[1][0].append(drugidx1)
                            edges_dst[1][0].append(drugidx2)
                            edges_src[1][0].append(drugidx2)
                            edges_dst[1][0].append(drugidx1)
                            edge_val[1][0].append(synergy)
                            edge_val[1][0].append(synergy)
                        elif (d1_fold == test_fold) or (d2_fold == test_fold):
                            edges_src[2][0].append(drugidx1)
                            edges_dst[2][0].append(drugidx2)
                            edges_src[2][0].append(drugidx2)
                            edges_dst[2][0].append(drugidx1)
                            edge_val[2][0].append(synergy)
                            edge_val[2][0].append(synergy)
                    elif (float(synergy) < upb) and (float(synergy) > lowb): #add
                        if (d1_fold in train_fold) and (d2_fold in train_fold):
                            edges_src[0][1].append(drugidx1)
                            edges_dst[0][1].append(drugidx2)
                            edges_src[0][1].append(drugidx2)
                            edges_dst[0][1].append(drugidx1)
                            edge_val[0][1].append(synergy)
                            edge_val[0][1].append(synergy)
                        elif (d1_fold == valid_fold) or (d2_fold == valid_fold):
                            edges_src[1][1].append(drugidx1)
                            edges_dst[1][1].append(drugidx2)
                            edges_src[1][1].append(drugidx2)
                            edges_dst[1][1].append(drugidx1)
                            edge_val[1][1].append(synergy)
                            edge_val[1][1].append(synergy)
                        elif (d1_fold == test_fold) or (d2_fold == test_fold):
                            edges_src[2][1].append(drugidx1)
                            edges_dst[2][1].append(drugidx2)
                            edges_src[2][1].append(drugidx2)
                            edges_dst[2][1].append(drugidx1)
                            edge_val[2][1].append(synergy)
                            edge_val[2][1].append(synergy)
                    elif float(synergy) < lowb:#ant
                        if (d1_fold in train_fold) and (d2_fold in train_fold):
                            edges_src[0][2].append(drugidx1)
                            edges_dst[0][2].append(drugidx2)
                            edges_src[0][2].append(drugidx2)
                            edges_dst[0][2].append(drugidx1)
                            edge_val[0][2].append(synergy)
                            edge_val[0][2].append(synergy)
                        elif (d1_fold == valid_fold) or (d2_fold == valid_fold):
                            edges_src[1][2].append(drugidx1)
                            edges_dst[1][2].append(drugidx2)
                            edges_src[1][2].append(drugidx2)
                            edges_dst[1][2].append(drugidx1)
                            edge_val[1][2].append(synergy)
                            edge_val[1][2].append(synergy)
                        elif (d1_fold == test_fold) or (d2_fold == test_fold):
                            edges_src[2][2].append(drugidx1)
                            edges_dst[2][2].append(drugidx2)
                            edges_src[2][2].append(drugidx2)
                            edges_dst[2][2].append(drugidx1)
                            edge_val[2][2].append(synergy)
                            edge_val[2][2].append(synergy)

                for i in range(self.drugscount):
                    for j in range(3):
                        edges_src[j][3].append(i)
                        edges_dst[j][3].append(i)
                        edge_val[j][3].append(0)
                

                train_graph_data = {}
                src = torch.LongTensor(edges_src[0][0])
                dst = torch.LongTensor(edges_dst[0][0])
                train_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][1])
                dst = torch.LongTensor(edges_dst[0][1])
                train_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][2])
                dst = torch.LongTensor(edges_dst[0][2])
                train_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[0][3])
                dst = torch.LongTensor(edges_dst[0][3])
                train_graph_data[('drug','intra','drug')] = (src, dst)

                train_graph = dgl.heterograph(train_graph_data)
                train_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[0][0]),('drug','add','drug'):torch.Tensor(edge_val[0][1]),('drug','ant','drug'):torch.Tensor(edge_val[0][2]),('drug','intra','drug'):torch.Tensor(edge_val[0][3])}

                train_g_list.append(train_graph)

                valid_graph_data = {}
                src = torch.LongTensor(edges_src[1][0])
                dst = torch.LongTensor(edges_dst[1][0])
                valid_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][1])
                dst = torch.LongTensor(edges_dst[1][1])
                valid_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][2])
                dst = torch.LongTensor(edges_dst[1][2])
                valid_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[1][3])
                dst = torch.LongTensor(edges_dst[1][3])
                valid_graph_data[('drug','intra','drug')] = (src, dst)

                valid_graph = dgl.heterograph(valid_graph_data)

                valid_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[1][0]),('drug','add','drug'):torch.Tensor(edge_val[1][1]),('drug','ant','drug'):torch.Tensor(edge_val[1][2]),('drug','intra','drug'):torch.Tensor(edge_val[1][3])}
                valid_g_list.append(valid_graph)

                test_graph_data = {}
                src = torch.LongTensor(edges_src[2][0])
                dst = torch.LongTensor(edges_dst[2][0])
                test_graph_data[('drug','syn','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][1])
                dst = torch.LongTensor(edges_dst[2][1])
                test_graph_data[('drug','add','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][2])
                dst = torch.LongTensor(edges_dst[2][2])
                test_graph_data[('drug','ant','drug')] = (src, dst)
                src = torch.LongTensor(edges_src[2][3])
                dst = torch.LongTensor(edges_dst[2][3])
                test_graph_data[('drug','intra','drug')] = (src, dst)

                test_graph = dgl.heterograph(test_graph_data)
                test_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[2][0]),('drug','add','drug'):torch.Tensor(edge_val[2][1]),('drug','ant','drug'):torch.Tensor(edge_val[2][2]),('drug','intra','drug'):torch.Tensor(edge_val[2][3])}
                test_g_list.append(test_graph)
        

            d = np.zeros([self.cellscount, self.drugscount, self.drugscount])
            edges_src = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            edges_dst = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            edge_val = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

            for each in self.data.values:
                drugname1, drugname2, cell_line, synergy, fold, d1_fold, d2_fold, c_fold = each
                cellidx = self.cellslist.index(cell_line)
                drugidx1 = self.drugslist.index(drugname1)
                drugidx2 = self.drugslist.index(drugname2)
                if (d1_fold in train_fold) and (d2_fold in train_fold):
                    if float(synergy) >= upb:
                        d[cellidx][drugidx1][drugidx2] = 1
                        d[cellidx][drugidx2][drugidx1] = 1
                    elif float(synergy) >= 0:
                        d[cellidx][drugidx1][drugidx2] = 2
                        d[cellidx][drugidx2][drugidx1] = 2
                    else:
                        d[cellidx][drugidx1][drugidx2] = 3
                        d[cellidx][drugidx2][drugidx1] = 3
                if float(synergy) >= upb: #syn
                    if (d1_fold in train_fold) and (d2_fold in train_fold):
                        edges_src[0][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][0].append(synergy)
                        edge_val[0][0].append(synergy)

                    elif (d1_fold == valid_fold) or (d2_fold == valid_fold):
                        edges_src[1][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][0].append(synergy)
                        edge_val[1][0].append(synergy)
                    elif (d1_fold == test_fold) or (d2_fold == test_fold):
                        edges_src[2][0].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][0].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][0].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][0].append(synergy)
                        edge_val[2][0].append(synergy)
                elif (float(synergy) > 0): #add
                    if (d1_fold in train_fold) and (d2_fold in train_fold):
                        edges_src[0][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][1].append(synergy)
                        edge_val[0][1].append(synergy)
                    elif (d1_fold == valid_fold) or (d2_fold == valid_fold):
                        edges_src[1][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][1].append(synergy)
                        edge_val[1][1].append(synergy)
                    elif (d1_fold == test_fold) or (d2_fold == test_fold):
                        edges_src[2][1].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][1].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][1].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][1].append(synergy)
                        edge_val[2][1].append(synergy)
                else:#ant
                    if (d1_fold in train_fold) and (d2_fold in train_fold):
                        edges_src[0][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[0][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[0][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[0][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[0][2].append(synergy)
                        edge_val[0][2].append(synergy)
                    elif (d1_fold == valid_fold) or (d2_fold == valid_fold):
                        edges_src[1][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[1][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[1][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[1][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[1][2].append(synergy)
                        edge_val[1][2].append(synergy)
                    elif (d1_fold == test_fold) or (d2_fold == test_fold):
                        edges_src[2][2].append(drugidx1 + cellidx * self.drugscount)
                        edges_dst[2][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_src[2][2].append(drugidx2 + cellidx * self.drugscount)
                        edges_dst[2][2].append(drugidx1 + cellidx * self.drugscount)
                        edge_val[2][2].append(synergy)
                        edge_val[2][2].append(synergy)


            for cid in range(self.cellscount * self.drugscount + self.cellscount + self.drugscount):
                for i in range(3):
                    edges_src[i][3].append(cid)
                    edges_dst[i][3].append(cid)
                    edge_val[i][3].append(0)

            '''
            universal: cell c*d + i
            drug  c*d+c + i
            '''

            
            for cid in range(self.cellscount):
                for did in range(self.drugscount):
                    edges_dst[0][3].append(self.cellscount * self.drugscount + cid)
                    edges_src[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    edges_src[0][3].append(self.cellscount * self.drugscount + cid)
                    edges_dst[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    
                    edges_dst[0][3].append(self.cellscount * self.drugscount + self.cellscount + did)
                    edges_src[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    edges_src[0][3].append(self.cellscount * self.drugscount + self.cellscount + did)
                    edges_dst[0][3].append(did + cid * self.drugscount)
                    edge_val[0][3].append(0)
                    

            for did in range(self.drugscount):
                for did2 in range(self.drugscount):
                    if did != did2:
                        edges_dst[0][3].append(self.cellscount * self.drugscount + cid + did)
                        edges_src[0][3].append(self.cellscount * self.drugscount + cid + did2)
                        edge_val[0][3].append(0)

            

            for i in range(self.cellscount):
                print(i)
                for j in range(self.cellscount):
                    for k in range(self.drugscount):
                        for l in range(k, self.drugscount):
                            if (i==j) and (k==l):
                                continue
                            union = 0
                            its = 0
                            for x in range(self.drugscount):
                                if (x == k) or (x == l):
                                    continue
                                if (d[i][k][x] == d[j][l][x]) and (d[i][k][x] != 0):
                                    union += 1
                                    its += 1
                                else:
                                    if (d[i][k][x] != 0):
                                        union += 1
                                    if (d[j][l][x] != 0):
                                        union += 1
                            if union == 0:
                                scr = 0.0
                            else:
                                scr = 1.0 * its / union
                            if scr >= 0.8:
                                edges_src[0][3].append(i * self.drugscount + k)
                                edges_dst[0][3].append(j * self.drugscount + l)
                                edges_src[0][3].append(j * self.drugscount + l)
                                edges_dst[0][3].append(i * self.drugscount + k)
                                edge_val[0][3].append(0)
                                edge_val[0][3].append(0)

            train_mask = []
            valid_mask = []
            test_mask = []

            train_graph_data = {}
            src = torch.LongTensor(edges_src[0][0])
            dst = torch.LongTensor(edges_dst[0][0])
            train_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[0][1])
            dst = torch.LongTensor(edges_dst[0][1])
            train_graph_data[('drug','add','drug')] = (src, dst)
            train_mask.append(torch.ones([len(edges_src[0][1])], dtype=torch.bool))
            #train_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[0][2])
            dst = torch.LongTensor(edges_dst[0][2])
            train_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[0][3])
            dst = torch.LongTensor(edges_dst[0][3])
            train_graph_data[('drug','intra','drug')] = (src, dst)
            
            train_graph = dgl.heterograph(train_graph_data)
            train_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[0][0]),('drug','add','drug'):torch.Tensor(edge_val[0][1]),('drug','ant','drug'):torch.Tensor(edge_val[0][2]),('drug','intra','drug'):torch.Tensor(edge_val[0][3])}

            #train_g_list.append(train_graph)

            valid_graph_data = {}
            src = torch.LongTensor(edges_src[1][0])
            dst = torch.LongTensor(edges_dst[1][0])
            valid_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[1][1])
            dst = torch.LongTensor(edges_dst[1][1])
            valid_graph_data[('drug','add','drug')] = (src, dst)
            valid_mask.append(torch.ones([len(edges_src[1][1])], dtype=torch.bool))
            #valid_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[1][2])
            dst = torch.LongTensor(edges_dst[1][2])
            valid_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[1][3])
            dst = torch.LongTensor(edges_dst[1][3])
            valid_graph_data[('drug','intra','drug')] = (src, dst)
            valid_graph = dgl.heterograph(valid_graph_data)

            valid_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[1][0]),('drug','add','drug'):torch.Tensor(edge_val[1][1]),('drug','ant','drug'):torch.Tensor(edge_val[1][2]),('drug','intra','drug'):torch.Tensor(edge_val[1][3])}
            #valid_g_list.append(valid_graph)

            test_graph_data = {}
            src = torch.LongTensor(edges_src[2][0])
            dst = torch.LongTensor(edges_dst[2][0])
            test_graph_data[('drug','syn','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[2][1])
            dst = torch.LongTensor(edges_dst[2][1])
            test_graph_data[('drug','add','drug')] = (src, dst)
            test_mask.append(torch.ones([len(edges_src[2][1])], dtype=torch.bool))
            #test_mask.append(torch.zeros([self.drugscount*self.cellscount], dtype=torch.bool))
            src = torch.LongTensor(edges_src[2][2])
            dst = torch.LongTensor(edges_dst[2][2])
            test_graph_data[('drug','ant','drug')] = (src, dst)
            src = torch.LongTensor(edges_src[2][3])
            dst = torch.LongTensor(edges_dst[2][3])
            test_graph_data[('drug','intra','drug')] = (src, dst)
            test_graph = dgl.heterograph(test_graph_data)
            test_graph.edata['syn'] = {('drug','syn','drug'):torch.Tensor(edge_val[2][0]),('drug','add','drug'):torch.Tensor(edge_val[2][1]),('drug','ant','drug'):torch.Tensor(edge_val[2][2]),('drug','intra','drug'):torch.Tensor(edge_val[2][3])}
            #test_g_list.append(test_graph)
            train_mask = torch.cat(train_mask)
            valid_mask = torch.cat(valid_mask)
            test_mask = torch.cat(test_mask)
            return train_g_list, valid_g_list, test_g_list, train_graph, valid_graph, test_graph,train_mask, valid_mask, test_mask#, train_g_list
        
