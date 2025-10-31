import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch import GraphConv
import dgl
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.conv import GATConv

class CAGATConv(nn.Module):
    def __init__(self, in_feats, cell_feats, num_heads, out_feats):
        super(CAGATConv, self).__init__()
        self.heads = num_heads
        self.out_feats = out_feats
        self.in_feats = in_feats
        self.cell_feats = cell_feats

        syn_conv = GATConv(in_feats + cell_feats, out_feats, num_heads, attn_drop=0.3)
        add_conv = GATConv(in_feats + cell_feats, out_feats, num_heads, attn_drop=0.3)
        ant_conv = GATConv(in_feats + cell_feats, out_feats, num_heads, attn_drop=0.3)
        intra_conv = GATConv(in_feats + cell_feats, out_feats, num_heads, attn_drop=0.3)

        self.conv = dgl.nn.HeteroGraphConv({
            'syn':syn_conv,
            'add':add_conv,
            'ant':ant_conv,
            'intra':intra_conv},
            aggregate='stack')

        self.wq = nn.Linear(cell_feats, out_feats)
        self.wk = nn.Linear(out_feats, out_feats)
        self.wv = nn.Linear(out_feats, out_feats)
        self.attn = nn.MultiheadAttention(out_feats, 1, batch_first=True)
        self.fc = nn.Linear(out_feats + cell_feats, out_feats)

    def forward(self, g, drug_feats, cell_feats):
        c_num = cell_feats.shape[0]
        d_num = (drug_feats.shape[0] - c_num) // (c_num + 1)
        #print(drug_feats.shape)
        cell_feat_pad = cell_feats.reshape(c_num, 1, -1).expand(c_num, d_num, -1).reshape(c_num * d_num, -1)
        cell_feat_pad2 = cell_feats.reshape(c_num, -1)
        cell_feat_pad3 = torch.mean(cell_feats.reshape(c_num, 1, -1).expand(c_num, d_num, -1), dim=0).reshape(d_num, -1)
        cell_feat_final = torch.cat([cell_feat_pad, cell_feat_pad2, cell_feat_pad3], 0).reshape(c_num * d_num + c_num + d_num, -1)
        in_feats = torch.cat((drug_feats, cell_feat_final), 1).reshape(c_num * d_num + c_num + d_num, -1)
        feats = {'drug': in_feats}
        rst = self.conv(g, feats)['drug']
        rst = rst.reshape(-1, self.heads * 4, self.out_feats) # c*d , 12, 512
        #rst = torch.mean(rst, 1)
        #rst = rst.reshape(c_num, d_num, -1)
        #cell_feat_Q = cell_feats.reshape(c_num, 1, -1).expand(c_num, d_num, -1).reshape(c_num * d_num, -1) # c * d,
        Q = self.wq(cell_feat_final).reshape(-1, 1, self.out_feats)
        K = self.wk(rst).reshape(-1, self.heads * 4, self.out_feats)
        V = self.wv(rst).reshape(-1, self.heads * 4, self.out_feats)
        attn_out, _ = self.attn(Q,K,V)
        attn_out = attn_out.reshape(c_num * d_num + c_num + d_num, -1)


        return attn_out

class MLPPredictor(nn.Module):
    def __init__(self, in_dims, cf_dims, orig_dims, o_cf_dims, n_cf_dims):
        super().__init__()
        self.drug_network1 = nn.Sequential(
            nn.Linear(orig_dims, orig_dims*2),
            nn.ReLU(),
            nn.Linear(orig_dims*2, orig_dims),
        )

        self.drug_network2 = nn.Sequential(
            nn.Linear(in_dims, in_dims*2),
            nn.ReLU(),
            nn.Linear(in_dims*2, in_dims),
        )

        self.drug_network3 = nn.Sequential(
            nn.Linear(in_dims, in_dims*2),
            nn.ReLU(),
            nn.Linear(in_dims*2, in_dims),
        )

        self.cell_network = nn.Sequential(
            nn.Linear(cf_dims, cf_dims * 2),
            nn.ReLU(),
            nn.Linear(cf_dims * 2, 256),
        )

        self.cell_network2 = nn.Sequential(
            nn.Linear(o_cf_dims, o_cf_dims),
            nn.ReLU(),
            nn.Linear(o_cf_dims, 256),
        )

        self.cell_network3 = nn.Sequential(
            nn.Linear(n_cf_dims, n_cf_dims * 2),
            nn.ReLU(),
            nn.Linear(n_cf_dims * 2, 256),
        )

        self.fc_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*(orig_dims + in_dims + in_dims)+ 256 * 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            #nn.Dropout(0.6),
            nn.Linear(256,1)
        )

    def apply_edges(self, edges):
        drug1_feat1_vector = self.drug_network1( edges.src['orig_h'] ) 
        drug1_feat2_vector = self.drug_network2( edges.src['spec_h'] )
        drug1_feat3_vector = self.drug_network3( edges.src['h'] )
        drug2_feat1_vector = self.drug_network1( edges.dst['orig_h'] ) 
        drug2_feat2_vector = self.drug_network2( edges.dst['spec_h'] )
        drug2_feat3_vector = self.drug_network3( edges.dst['h'] )
        cell_feat_vector = self.cell_network(edges.src['cf'])
        cell_feat_vector2 = self.cell_network2(edges.src['cfo'])
        cell_feat_vector3 = self.cell_network3(edges.src['cfn'])
        # cell_feat_vector = cell_feat
        feat = torch.cat([drug1_feat1_vector, drug1_feat2_vector,drug1_feat3_vector , drug2_feat1_vector, drug2_feat2_vector, drug2_feat3_vector, cell_feat_vector, cell_feat_vector2, cell_feat_vector3], 1)
        #feat2 = torch.cat([drug2_feat1_vector, drug2_feat2_vector,drug2_feat3_vector , drug1_feat1_vector, drug1_feat2_vector, drug1_feat3_vector, cell_feat_vector, cell_feat_vector2, cell_feat_vector3], 1)
        out = self.fc_network(feat)
        out = out.reshape(-1)
        return {'score': out}

    def forward(self, graph, h, cell_feat, spec_h, orig_h, orig_cf, net_cf):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['spec_h'] = spec_h.reshape(graph.num_nodes(), -1)
            graph.ndata['orig_h'] = orig_h
            expand_cell_feat = cell_feat.reshape(1, -1).expand(graph.num_nodes(), -1)
            expand_cell_feat_o = orig_cf.reshape(1, -1).expand(graph.num_nodes(), -1)
            expand_cell_feat_n = net_cf.reshape(1, -1).expand(graph.num_nodes(), -1)
            graph.ndata['cf'] = expand_cell_feat # cell feat
            graph.ndata['cfo'] = expand_cell_feat_o # cell feat
            graph.ndata['cfn'] = expand_cell_feat_n # cell feat
            graph.apply_edges(self.apply_edges, etype='syn')#, edges=edge_mask)
            graph.apply_edges(self.apply_edges, etype='add')
            graph.apply_edges(self.apply_edges, etype='ant')
            
            return graph.edata['score']

class CATSynModel(nn.Module):
    def __init__(self, in_feats, cell_feats, cell_hidden, num_heads, out_feats):
        super(CAGSynModel, self).__init__()
        self.heads = num_heads
        self.out_feats = out_feats
        self.in_feats = in_feats
        self.cell_feats = cell_feats
        self.cell_hidden = cell_hidden

        self.hetero_trans = nn.Linear(cell_feats, in_feats)

        self.cf_net = nn.Linear(cell_feats, cell_hidden)

        self.conv1 = CAGATConv(in_feats, cell_hidden, num_heads, out_feats)
        self.conv2 = CAGATConv(out_feats, cell_hidden, num_heads, out_feats)
        self.pred = MLPPredictor(out_feats, cell_hidden, in_feats, cell_feats, out_feats)
        self.edge_type = ['syn','add','ant','intra']

    def forward(self, g, dec_graph, x, cf):
        cf_vec = self.cf_net(cf)

        d_num = x.shape[0]
        c_num = cf.shape[0]
        d_feat_pad = x.reshape(1, d_num, -1).expand(c_num, d_num, -1).reshape(c_num * d_num, -1)
        d_feat_ud = x
        d_feat_uc = self.hetero_trans(cf)
        d_feat = torch.cat([d_feat_pad, d_feat_uc, d_feat_ud], 0).reshape(c_num * d_num + c_num + d_num, -1)
        h = self.conv1(g, d_feat, cf_vec)
        h = self.conv2(g, h, cf_vec)
        return h

    def inference(self, g, x, cf):
        cf_vec = self.cf_net(cf)

        d_num = x.shape[0]
        c_num = cf.shape[0]
        d_feat_pad = x.reshape(1, d_num, -1).expand(c_num, d_num, -1).reshape(c_num * d_num, -1)
        d_feat_ud = x
        d_feat_uc = self.hetero_trans(cf)
        d_feat = torch.cat([d_feat_pad, d_feat_uc, d_feat_ud], 0).reshape(c_num * d_num + c_num + d_num, -1)
        h = self.conv1(g, d_feat, cf_vec)
        h = self.conv2(g, h, cf_vec)
        return h

    def predict(self, g, x, cf, x_spec, x_orig, ncf):
        cf_vec = self.cf_net(cf)
        res = self.pred(g, x, cf_vec, x_spec, x_orig, cf, ncf)
        return res

