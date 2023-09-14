import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.batch import unbatch
from dgl.transforms import shortest_dist
from torch_geometric.nn import EGConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

from param import *

args = parse_args()


class DegreeEncoder(nn.Module):
    def __init__(self, max_degree, embedding_dim):
        super(DegreeEncoder, self).__init__()
        self.encoder1 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.encoder2 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.max_degree = max_degree

    def forward(self, g):
        in_degree = th.clamp(g.in_degrees(), min=0, max=self.max_degree)
        out_degree = th.clamp(g.out_degrees(), min=0, max=self.max_degree)
        degree_embedding = self.encoder1(in_degree) + self.encoder2(out_degree)
        return degree_embedding


class SpatialEncoder(nn.Module):
    def __init__(self, max_dist, num_heads=1):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_heads, padding_idx=0
        )

    def forward(self, g):
        device = g.device
        g_list = unbatch(g)
        max_num_nodes = th.max(g.batch_num_nodes())
        spatial_encoding = th.zeros(
            len(g_list), max_num_nodes, max_num_nodes, self.num_heads
        ).to(device)

        for i, ubg in enumerate(g_list):
            num_nodes = ubg.num_nodes()
            dist = (
                    th.clamp(
                        shortest_dist(ubg, root=None, return_paths=False),
                        min=-1,
                        max=self.max_dist,
                    )
                    + 1
            )
            dist_embedding = self.embedding_table(dist)
            spatial_encoding[i, :num_nodes, :num_nodes] = dist_embedding
        return spatial_encoding


class BiasedMHA(nn.Module):

    def __init__(
            self,
            feat_size,
            num_heads,
            bias=True,
            attn_bias_type="add",
            attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
                self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.u_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.GLU = nn.Linear(feat_size, feat_size, bias=bias)

        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.u_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.GLU.weight, gain=2 ** -0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, attn_bias=None, attn_mask=None):

        q_h = self.q_proj(ndata).transpose(0, 1)
        k_h = self.k_proj(ndata).transpose(0, 1)
        v_h = self.v_proj(ndata).transpose(0, 1)
        # u_h = self.u_proj(ndata)
        u_h = ndata
        u_h = u_h * torch.sigmoid(self.GLU(u_h))
        bsz, N, _ = ndata.shape
        q_h = (
                q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)
                * self.scaling
        )
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(
            1, 2, 0
        )
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )

        attn_weights = (
            th.bmm(q_h, k_h)
                .transpose(0, 2)
                .reshape(N, N, bsz, self.num_heads)
                .transpose(0, 2)
        )

        if attn_bias is not None:
            if self.attn_bias_type == "add":
                attn_weights += attn_bias
            else:
                attn_weights *= attn_bias
        if attn_mask is not None:
            attn_weights[attn_mask.to(th.bool)] = float("-inf")
        attn_weights = F.softmax(
            attn_weights.transpose(0, 2)
                .reshape(N, N, bsz * self.num_heads)
                .transpose(0, 2),
            dim=2,
        )

        attn_weights = self.dropout(attn_weights)

        attn = th.bmm(attn_weights, v_h).transpose(0, 1)

        attn = self.out_proj(
            attn.reshape(N, bsz, self.feat_size).transpose(0, 1)
        )
        attn = u_h + attn
        return attn


class GraphormerLayer(nn.Module):

    def __init__(
            self,
            feat_size,
            hidden_size,
            num_heads,
            attn_bias_type="add",
            norm_first=False,
            dropout=0.1,
            attn_dropout=0.1,
            activation=nn.ReLU(),
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = BiasedMHA(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=attn_dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, feat_size),
            nn.Dropout(p=dropout),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, attn_bias=None, attn_mask=None):

        residual = nfeat
        if self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        nfeat = self.attn(nfeat, attn_bias, attn_mask)
        nfeat = self.dropout(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = self.ffn(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        return nfeat


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(args.hidden, args.hidden // 2)
        self.bn1 = nn.BatchNorm1d(args.hidden // 2)
        self.dropout1 = nn.Dropout(args.MLPDropout)
        self.fc2 = nn.Linear(args.hidden // 2, args.hidden // 4)
        self.bn2 = nn.BatchNorm1d(args.hidden // 4)
        self.dropout2 = nn.Dropout(args.MLPDropout)
        self.fc3 = nn.Linear(args.hidden // 4, args.hidden // 8)
        self.bn3 = nn.BatchNorm1d(args.hidden // 8)
        self.dropout3 = nn.Dropout(args.MLPDropout)
        self.fc4 = nn.Linear(args.hidden // 8, 1)

    def forward(self, x):
        x = F.tanh(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.tanh(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.tanh(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class GG_transfor(nn.Module):
    def __init__(self, args, n_rna, n_dis):
        super(GG_transfor, self).__init__()
        self.degree_encoder = DegreeEncoder(8, args.hidden)
        self.spatial_encoder = SpatialEncoder(max_dist=8, num_heads=8)
        self.graphormer_layer = GraphormerLayer(
            feat_size=args.hidden,
            hidden_size=2048,
            num_heads=8
        )

    def forward(self, newFeature, graph):
        degree_embedding = self.degree_encoder(graph)

        newFeature = newFeature + degree_embedding
        newFeature = newFeature.unsqueeze(0)

        spatial_embedding = self.spatial_encoder(graph)
        bias = spatial_embedding

        out = self.graphormer_layer(newFeature, bias)
        out = out.squeeze(0)

        return out


class Feature_Preprocessing(nn.Module):
    def __init__(self, args, n_rna, n_dis):
        super(Feature_Preprocessing, self).__init__()

        self.m_g_rnn = nn.LSTM(n_rna, args.hidden, 2)
        self.d_g_rnn = nn.LSTM(n_dis, args.hidden, 2)
        self.fc_m_g = nn.Linear(args.hidden, args.hidden)
        self.fc_d_g = nn.Linear(args.hidden, args.hidden)
        self.h0_m_g = torch.randn(2, args.hidden).to(device=args.device)
        self.c0_m_g = torch.randn(2, args.hidden).to(device=args.device)
        self.h1_d_g = torch.randn(2, args.hidden).to(device=args.device)
        self.c1_d_g = torch.randn(2, args.hidden).to(device=args.device)

        self.m_f_rnn = nn.LSTM(n_rna, args.hidden, 2)
        self.d_s_rnn = nn.LSTM(n_dis, args.hidden, 2)
        self.fc_m_f = nn.Linear(args.hidden, args.hidden)
        self.fc_d_s = nn.Linear(args.hidden, args.hidden)
        self.h0_m_f = torch.randn(2, args.hidden).to(device=args.device)
        self.c0_m_f = torch.randn(2, args.hidden).to(device=args.device)
        self.h1_d_s = torch.randn(2, args.hidden).to(device=args.device)
        self.c1_d_s = torch.randn(2, args.hidden).to(device=args.device)

        self.gongxiangM1 = nn.Linear(n_rna, args.hidden)
        self.gongxiangD1 = nn.Linear(n_dis, args.hidden)

    def forward(self, args, m_f, d_f):
        m_g_feature = m_f[0]
        d_g_feature = d_f[0]
        m_f_feature = m_f[1]
        d_s_feature = d_f[1]

        m_g_f, (hn, cn) = self.m_g_rnn(m_g_feature, (self.h0_m_g, self.c0_m_g))
        d_g_f, (hn, cn) = self.d_g_rnn(d_g_feature, (self.h1_d_g, self.c1_d_g))
        m_g_f = self.fc_m_g(m_g_f)
        d_g_f = self.fc_d_g(d_g_f)

        m_f_f, (hn, cn) = self.m_f_rnn(m_f_feature, (self.h0_m_f, self.c0_m_f))
        d_s_f, (hn, cn) = self.d_s_rnn(d_s_feature, (self.h1_d_s, self.c1_d_s))
        m_f_f = self.fc_m_f(m_f_f)
        d_s_f = self.fc_d_s(d_s_f)

        canchaM_g1 = self.gongxiangM1(m_g_feature)
        canchaM_f1 = self.gongxiangM1(m_f_feature)
        canchaD_g1 = self.gongxiangD1(d_g_feature)
        canchaD_s1 = self.gongxiangD1(d_s_feature)

        shuchuM_g = canchaM_g1 * m_g_f + m_g_f
        shuchuM_f = canchaM_f1 * m_f_f + m_f_f
        shuchuD_g = canchaD_g1 * d_g_f + d_g_f
        shuchuD_s = canchaD_s1 * d_s_f + d_s_f

        MF = shuchuM_g + shuchuM_f
        DF = shuchuD_g + shuchuD_s
        x = torch.cat([MF, DF], dim=0)
        return x


class Feature_MetaLST(nn.Module):
    def __init__(self, args):
        super(Feature_MetaLST, self).__init__()

        self.m_g_rnn = nn.LSTM(args.hidden, args.hidden, 2)
        self.d_g_rnn = nn.LSTM(args.hidden, args.hidden, 2)
        self.fc_m_g = nn.Linear(args.hidden, args.hidden)
        self.fc_d_g = nn.Linear(args.hidden, args.hidden)
        self.h0_m_g = torch.randn(2, args.hidden).to(device=args.device)
        self.c0_m_g = torch.randn(2, args.hidden).to(device=args.device)
        self.h1_d_g = torch.randn(2, args.hidden).to(device=args.device)
        self.c1_d_g = torch.randn(2, args.hidden).to(device=args.device)

        self.m_f_rnn = nn.LSTM(args.hidden, args.hidden, 2)
        self.d_s_rnn = nn.LSTM(args.hidden, args.hidden, 2)
        self.fc_m_f = nn.Linear(args.hidden, args.hidden)
        self.fc_d_s = nn.Linear(args.hidden, args.hidden)
        self.h0_m_f = torch.randn(2, args.hidden).to(device=args.device)
        self.c0_m_f = torch.randn(2, args.hidden).to(device=args.device)
        self.h1_d_s = torch.randn(2, args.hidden).to(device=args.device)
        self.c1_d_s = torch.randn(2, args.hidden).to(device=args.device)

        self.gongxiangM1 = nn.Linear(args.hidden, args.hidden)
        self.gongxiangD1 = nn.Linear(args.hidden, args.hidden)

    def forward(self, m_g_feature, m_f_feature, d_g_feature, d_s_feature):
        m_g_f, (hn, cn) = self.m_g_rnn(m_g_feature, (self.h0_m_g, self.c0_m_g))
        d_g_f, (hn, cn) = self.d_g_rnn(d_g_feature, (self.h1_d_g, self.c1_d_g))
        m_g_f = self.fc_m_g(m_g_f)
        d_g_f = self.fc_d_g(d_g_f)

        m_f_f, (hn, cn) = self.m_f_rnn(m_f_feature, (self.h0_m_f, self.c0_m_f))
        d_s_f, (hn, cn) = self.d_s_rnn(d_s_feature, (self.h1_d_s, self.c1_d_s))
        m_f_f = self.fc_m_f(m_f_f)
        d_s_f = self.fc_d_s(d_s_f)

        canchaM_g1 = self.gongxiangM1(m_g_feature)
        canchaM_f1 = self.gongxiangM1(m_f_feature)
        canchaD_g1 = self.gongxiangD1(d_g_feature)
        canchaD_s1 = self.gongxiangD1(d_s_feature)

        shuchuM_g = canchaM_g1 * m_g_f + m_g_f
        shuchuM_f = canchaM_f1 * m_f_f + m_f_f
        shuchuD_g = canchaD_g1 * d_g_f + d_g_f
        shuchuD_s = canchaD_s1 * d_s_f + d_s_f

        MF = shuchuM_g + shuchuM_f
        DF = shuchuD_g + shuchuD_s
        return MF, DF


class Feature_PreLinear(nn.Module):
    def __init__(self, args, n_rna, n_dis):
        super(Feature_PreLinear, self).__init__()

        self.mf = nn.Linear(n_rna, args.hidden)
        self.df = nn.Linear(n_dis, args.hidden)

    def forward(self, args, m_f, d_f):
        m_f = self.mf(m_f)
        d_f = self.df(d_f)
        return m_f, d_f


class MY_GNN(nn.Module):
    def __init__(self, args, hidden_channels, num_layers, num_heads):
        super(MY_GNN, self).__init__()
        aggregators = ['symnorm']
        self.fc = nn.Linear(hidden_channels * (num_layers + 1), args.hidden)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregators, num_heads)
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, args, data, edge_index):

        JK = []
        JK.append(data)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(data, edge_index)
            h = norm(h)
            h = h.relu_()
            JK.append(h)

        ALL = torch.cat(JK, dim=-1)
        res = self.fc(ALL)
        return res


class XXYY2F(nn.Module):
    def __init__(self, args):
        super(XXYY2F, self).__init__()
        self.fc = nn.Linear(args.hidden * 2, args.hidden)
        self.relu = nn.ReLU();

    def forward(self, XXYY_X1, XXYY_X2, XXYY_Y1, XXYY_Y2):
        average_X = torch.mean(torch.stack([XXYY_X1, XXYY_X2], dim=0), dim=0)
        average_Y = torch.mean(torch.stack([XXYY_Y1, XXYY_Y2], dim=0), dim=0)
        merged_XY = torch.cat([average_X, average_Y], dim=1)
        meta_XXYY = self.relu(self.fc(merged_XY))
        return meta_XXYY


class XYXY2F(nn.Module):
    def __init__(self, args):
        super(XYXY2F, self).__init__()

        self.fc1 = nn.Linear(args.hidden * 2, args.hidden)
        self.fc2 = nn.Linear(args.hidden * 2, args.hidden)
        self.relu = nn.ReLU()

    def forward(self, XYXY_X1, XYXY_Y1, XYXY_X2, XYXY_Y2):
        merged_XY1 = torch.cat([XYXY_X1, XYXY_Y1], dim=1)
        merged_XY2 = torch.cat([XYXY_X2, XYXY_Y2], dim=1)
        XY1 = self.relu(self.fc1(merged_XY1))
        XY2 = self.relu(self.fc2(merged_XY2))
        meta_XYXY = torch.mean(torch.stack([XY1, XY2], dim=0), dim=0)
        return meta_XYXY


class GAT_LP(nn.Module):
    def __init__(self, args):
        super(GAT_LP, self).__init__()
        head = 8
        self.ELU = nn.ELU()
        self.gat1 = GATv2Conv(args.hidden, args.hidden // 2, head, concat=True, dropout=0.1)
        self.gat2 = GATv2Conv(args.hidden // 2 * head, args.hidden // 4, head, concat=True, dropout=0.1)
        self.JK = JumpingKnowledge('cat')

        self.JKLin = nn.Linear(args.hidden + args.hidden // 2 * head + args.hidden // 4 * head, args.hidden)

    def forward(self, x, edge_index):
        JKList = []
        JKList.append(x)

        x = self.gat1(x, edge_index)
        x = self.ELU(x)
        JKList.append(x)
        x = self.gat2(x, edge_index)
        x = self.ELU(x)
        JKList.append(x)
        # x = self.gat3(x, edge_index)
        # x = self.ELU(x)
        # JKList.append(x)
        # x = self.gat4(x, edge_index)
        # x = self.ELU(x)
        # JKList.append(x)

        x = self.JK(JKList)
        x = self.JKLin(x)
        return x


class Meta_Fuse(nn.Module):
    def __init__(self, args):
        super(Meta_Fuse, self).__init__()
        self.mmdd_F = XXYY2F(args)
        self.ddmm_F = XXYY2F(args)
        self.mdmd_F = XYXY2F(args)
        self.dmdm_F = XYXY2F(args)

        self.Feature_MetaLST = Feature_MetaLST(args)

    def forward(self, mf, df, mmdd, ddmm, mdmd, dmdm):
        mmdd_m1, mmdd_m2, mmdd_d1, mmdd_d2 = self.fuseF1(mf, df, mmdd)
        ddmm_d1, ddmm_d2, ddmm_m1, ddmm_m2 = self.fuseF1(df, mf, ddmm)
        mdmd_m1, mdmd_d1, mdmd_m2, mdmdd_d2 = self.fuseF2(mf, df, mdmd)
        dmdm_d1, dmdm_m1, dmdm_d2, dmdm_m2 = self.fuseF2(df, mf, dmdm)
        mmdd_F = self.mmdd_F(mmdd_m1, mmdd_m2, mmdd_d1, mmdd_d2)
        ddmm_F = self.ddmm_F(ddmm_d1, ddmm_d2, ddmm_m1, ddmm_m2)
        mdmd_F = self.mdmd_F(mdmd_m1, mdmd_d1, mdmd_m2, mdmdd_d2)
        dmdm_F = self.dmdm_F(dmdm_d1, dmdm_m1, dmdm_d2, dmdm_m2)

        Meta_M, Meta_D = self.Feature_MetaLST(mmdd_F, mdmd_F, ddmm_F, dmdm_F)

        return Meta_M, Meta_D

    def fuseF1(self, X, Y, XXYY):

        dimXx, dimXy = X.shape

        # 沿着最后一个维度拆分成四个 [20, 495] 的张量
        split_tensors = torch.split(XXYY, 1, dim=-1)
        split_tensors = [t.squeeze(-1) for t in split_tensors]

        # 将每个 [20, 495] 的张量再拆分成 20 个 [495] 的张量
        final_split_tensors = []
        for split_tensor in split_tensors:
            sub_split_tensors = torch.split(split_tensor, 1, dim=0)
            sub_split_tensors = [t.squeeze(0) for t in sub_split_tensors]
            final_split_tensors.extend(sub_split_tensors)

        XXYY_X1 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XXYY_X1[i] = XXYY_X1[i] + X.index_select(dim=0, index=final_split_tensors[i].to(args.device))
        XXYY_X1, _ = torch.max(XXYY_X1, dim=0)

        XXYY_X2 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XXYY_X2[i] = XXYY_X2[i] + X.index_select(dim=0, index=final_split_tensors[i + 20].to(args.device))
        XXYY_X2, _ = torch.max(XXYY_X2, dim=0)

        XXYY_Y1 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XXYY_Y1[i] = XXYY_Y1[i] + Y.index_select(dim=0, index=final_split_tensors[i + 40].to(args.device))
        XXYY_Y1, _ = torch.max(XXYY_Y1, dim=0)

        XXYY_Y2 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XXYY_Y2[i] = XXYY_Y2[i] + Y.index_select(dim=0, index=final_split_tensors[i + 60].to(args.device))
        XXYY_Y2, _ = torch.max(XXYY_Y2, dim=0)
        return XXYY_X1, XXYY_X2, XXYY_Y1, XXYY_Y2

    def fuseF2(self, X, Y, XYXY):

        dimXx, dimXy = X.shape

        # 沿着最后一个维度拆分成四个 [20, 495] 的张量
        split_tensors = torch.split(XYXY, 1, dim=-1)
        split_tensors = [t.squeeze(-1) for t in split_tensors]

        # 将每个 [20, 495] 的张量再拆分成 20 个 [495] 的张量
        final_split_tensors = []
        for split_tensor in split_tensors:
            sub_split_tensors = torch.split(split_tensor, 1, dim=0)
            sub_split_tensors = [t.squeeze(0) for t in sub_split_tensors]
            final_split_tensors.extend(sub_split_tensors)

        XYXY_X1 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XYXY_X1[i] = XYXY_X1[i] + X.index_select(dim=0, index=final_split_tensors[i].to(args.device))
        XYXY_X1, _ = torch.max(XYXY_X1, dim=0)

        XYXY_Y1 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XYXY_Y1[i] = XYXY_Y1[i] + Y.index_select(dim=0, index=final_split_tensors[i + 20].to(args.device))
        XYXY_Y1, _ = torch.max(XYXY_Y1, dim=0)

        XYXY_X2 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XYXY_X2[i] = XYXY_X2[i] + X.index_select(dim=0, index=final_split_tensors[i + 40].to(args.device))
        XYXY_X2, _ = torch.max(XYXY_X2, dim=0)

        XYXY_Y2 = torch.zeros(20, dimXx, dimXy, device=args.device)
        for i in range(0, 20):
            XYXY_Y2[i] = XYXY_Y2[i] + Y.index_select(dim=0, index=final_split_tensors[i + 60].to(args.device))
        XYXY_Y2, _ = torch.max(XYXY_Y2, dim=0)
        return XYXY_X1, XYXY_Y1, XYXY_X2, XYXY_Y2


class MY_Module(nn.Module):  # 0.951  0.9494
    def __init__(self, args, n_rna, n_dis):
        super(MY_Module, self).__init__()
        self.n_rna = n_rna
        self.n_dis = n_dis

        self.Feature_PreLinear = Feature_PreLinear(args, n_rna, n_dis)
        self.Meta_Fuse = Meta_Fuse(args)
        self.GAT_LP = GAT_LP(args)
        self.GG_transfor = GG_transfor(args, n_rna, n_dis)
        self.GNN = MY_GNN(args, 512, 3, 4)
        self.MLP = MLP(args)

        self.attention_matrix = nn.Parameter(torch.ones(n_rna + n_dis, n_rna + n_dis))
        self.bn1 = nn.BatchNorm1d(args.hidden)
        self.bn2 = nn.BatchNorm1d(args.hidden)

    def encode(self, args, similarity_feature, graph, edge_idx_dict, i):
        m_f = similarity_feature['m_s']['Data_M']
        d_f = similarity_feature['d_s']['Data_M']

        m_d_graph = graph[str(i)]["fold_train_edges_80p_80n"]

        mf, df = self.Feature_PreLinear(args, m_f, d_f)
        ORIF = torch.cat([mf, df], dim=0)

        out = self.GG_transfor(ORIF, m_d_graph)

        ORIF = self.bn1(ORIF)
        out = out + ORIF

        Trans_miRNA = out[:self.n_rna, :]
        Trans_disease = out[self.n_rna:, :]

        Meta_M, Meta_D = self.Meta_Fuse(Trans_miRNA, Trans_disease,
                                        graph[str(i)]['train_mmdd_meta'], graph[str(i)]['train_ddmm_meta'],
                                        graph[str(i)]['train_mdmd_meta'], graph[str(i)]['train_dmdm_meta'])

        META = torch.cat([Meta_M, Meta_D], dim=0)

        META = self.bn2(META)
        META = out + META

        einx = edge_idx_dict[str(i)]['fold_train_edges_80p_80n']
        edge_index = to_undirected(einx)
        sumNode = self.n_rna + self.n_dis
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(sumNode, sumNode))
        res = self.GNN(args, META, adj)
        return res

    def decode(self, out, edge_label_index, i):
        miRNA_embedding = out[edge_label_index[0]]
        disease_embedding = out[edge_label_index[1]]
        res = (miRNA_embedding * disease_embedding)
        res = self.MLP(res)
        return res

    def forward(self, args, similarity_feature, graph, edge_idx_dict, edge_label_index, i):
        out = self.encode(args, similarity_feature, graph, edge_idx_dict, i)
        res = self.decode(out, edge_label_index, i)
        return res
