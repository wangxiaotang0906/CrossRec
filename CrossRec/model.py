import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool, TransformerConv, MessagePassing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def binary(x: torch.Tensor, k: float) -> torch.Tensor:
    return 0 if x < k else 1

def ste_binary(x, threshold=1.0):
    binary = (x > threshold).float()
    return binary.detach() + x - x.detach()

class NodeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5):
        super(NodeEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        return x

class BiologicalDrivenGraphTransformer(MessagePassing):
    def __init__(self, rna_in_channels, atac_in_channels, hidden_channels, num_heads=1, dropout=0.5):
        super(BiologicalDrivenGraphTransformer, self).__init__(aggr='mean')
        assert hidden_channels % num_heads == 0, "The number of hidden channels must be divisible by the number of heads."

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.hidden_channels = hidden_channels
        print(f'num_heads:{self.num_heads},head_dims:{self.head_dim}')

        # LayerNorm
        # self.input_rna_norm = nn.LayerNorm(rna_in_channels)
        # self.input_atac_norm = nn.LayerNorm(atac_in_channels)
        # self.layernorm_rna = nn.LayerNorm(hidden_channels)
        # self.layernorm_atac = nn.LayerNorm(hidden_channels)

        # Biological-driven crossmodal attention QKV
        self.rna_query = nn.Linear(rna_in_channels, hidden_channels)
        # self.atac_query = nn.Linear(atac_in_channels, hidden_channels)
        # self.rna_key = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_key = nn.Linear(atac_in_channels, hidden_channels)
        self.rna_value = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_value = nn.Linear(atac_in_channels, hidden_channels)

        # Output layer linear mapping
        if self.num_heads>1:
            self.rna_out_proj = nn.Linear(hidden_channels, hidden_channels)
            self.atac_out_proj = nn.Linear(hidden_channels, hidden_channels)

        # Intra-information
        self.rna_self_attention = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_self_attention = nn.Linear(atac_in_channels, hidden_channels)

        # Dim reduction
        self.dim_reduction_rna = nn.Linear(2 * hidden_channels, hidden_channels)
        self.dim_reduction_atac = nn.Linear(2 * hidden_channels, hidden_channels)

        # FFN
        # self.ffn_rna = nn.Sequential(
        #     nn.Linear(hidden_channels, 2 * hidden_channels),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(2 * hidden_channels, hidden_channels),
        #     nn.Dropout(dropout)
        # )
        # self.ffn_atac = nn.Sequential(
        #     nn.Linear(hidden_channels, 2 * hidden_channels),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(2 * hidden_channels, hidden_channels),
        #     nn.Dropout(dropout)
        # )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Counter, for case study
        self.sample_counter = 0

    def forward(self, x_rna, x_atac, chrom_mask, generation = False, save_attn_flag = False, save_dir = "./"):
        """
        x_rna: [num_rna_nodes, rna_in_channels]
        x_atac: [num_atac_nodes, atac_in_channels]
        """
        # Biologically-driven crossmodal attention
        rna_query = self.rna_query(x_rna).view(-1, self.num_heads, self.head_dim)  # [num_rna_nodes, num_heads, head_dim]
        atac_key = self.atac_key(x_atac).view(-1, self.num_heads, self.head_dim)  # [num_atac_nodes, num_heads, head_dim]
        atac_value = self.atac_value(x_atac).view(-1, self.num_heads, self.head_dim)  # [num_atac_nodes, num_heads, head_dim]
        rna_value = self.rna_value(x_rna).view(-1, self.num_heads, self.head_dim)  # [num_rna_nodes, num_heads, head_dim]
        rna_atac_attention = torch.bmm(
            rna_query.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim),
            atac_key.permute(1, 2, 0).reshape(self.num_heads, self.head_dim, -1)
        )  # [num_heads, num_rna_nodes, num_atac_nodes]

        #  Non-trainable heuristic generation module
        if generation == True:
            rna_atac_attention = ste_binary(torch.sigmoid(rna_atac_attention), threshold=0.8)
            attn_per_rna = rna_atac_attention.sum(dim=2)  # [N_rna]
            generated_rna  = ste_binary(attn_per_rna, threshold=9)
            return generated_rna
        else:
            chrom_mask = chrom_mask.squeeze(-1)
            rna_atac_attention = rna_atac_attention * chrom_mask

        # softmax
        # rna_atac_attention = F.softmax(rna_atac_attention, dim=-1)

        # binary
        # rna_atac_attention = torch.sigmoid(rna_atac_attention)
        # rna_atac_attention = (rna_atac_attention > 0.8).float()

        # Biological pruning strategy (Threshold + Top-K)
        rna_atac_attention = torch.sigmoid(rna_atac_attention)
        threshold_mask = (rna_atac_attention > 0.8).float()  # [num_heads, num_rna_nodes, 1]
        topk_values, topk_indices = torch.topk(rna_atac_attention, k=10, dim=-1)  # [num_heads, num_rna_nodes, top_k]
        topk_values = F.softmax(topk_values, dim=-1)
        rna_atac_attention = torch.zeros_like(rna_atac_attention)  # [num_heads, num_rna_nodes, num_atac_nodes]
        rna_atac_attention.scatter_(-1, topk_indices, topk_values)
        rna_atac_attention = rna_atac_attention * threshold_mask
        atac_rna_attention = rna_atac_attention.permute(0, 2, 1)

        # Saving attention martix, for case study
        if (save_attn_flag) and (self.training==False):
            save_dir_attention = os.path.join(save_dir, "rna_atac_attention")
            os.makedirs(save_dir_attention, exist_ok=True)
            sample_id = self.sample_counter
            self.sample_counter += 1
            save_path = os.path.join(save_dir_attention, f"sample_{sample_id}.pt")
            torch.save(rna_atac_attention.cpu(), save_path)
            print(f"Saved attention matrix for sample {sample_id} at {save_path}")

        # Aggregation
        rna_to_atac_agg = torch.bmm(rna_atac_attention, atac_value.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim))  # [num_heads, num_rna_nodes, head_dim]
        rna_to_atac_agg = rna_to_atac_agg.permute(1, 0, 2).reshape(-1, self.hidden_channels)  # [num_rna_nodes, hidden_channels]
        atac_to_rna_agg = torch.bmm(atac_rna_attention, rna_value.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim))  # [num_heads, num_atac_nodes, head_dim]
        atac_to_rna_agg = atac_to_rna_agg.permute(1, 0, 2).reshape(-1, self.hidden_channels)  # [num_atac_nodes, hidden_channels]

        if self.num_heads > 1:
            rna_to_atac_agg = self.rna_out_proj(rna_to_atac_agg)  # [num_rna_nodes, hidden_channels]
            atac_to_rna_agg = self.atac_out_proj(atac_to_rna_agg)  # [num_atac_nodes, hidden_channels]

        # Message Passing Aggregation（inter-information and intra-information）
        rna_to_atac_agg = torch.cat([rna_to_atac_agg, self.rna_self_attention(x_rna)], dim=1)
        atac_to_rna_agg = torch.cat([atac_to_rna_agg, self.atac_self_attention(x_atac)], dim=1)
        rna_to_atac_agg = self.dim_reduction_rna(rna_to_atac_agg)
        atac_to_rna_agg = self.dim_reduction_atac(atac_to_rna_agg)

        # Norm
        # rna_to_atac_agg = self.layernorm_rna(rna_to_atac_agg)  # [num_rna_nodes, hidden_channels]
        # atac_to_rna_agg = self.layernorm_atac(atac_to_rna_agg)  # [num_atac_nodes, hidden_channels]

        # FFN
        # rna_to_atac_agg = self.layernorm_rna(rna_to_atac_agg + self.ffn_rna(rna_to_atac_agg))  # FFN
        # atac_to_rna_agg = self.layernorm_atac(atac_to_rna_agg + self.ffn_atac(atac_to_rna_agg))  # FFN

        return rna_to_atac_agg, atac_to_rna_agg

class CrossRec(nn.Module):
    def __init__(self, in_channels_dict, id_embedding_dims, hidden_channels, out_channels, rna_database_path, generation=False, save_attn_flag=False, save_dir = "./", layer_num=1, multihead=True, num_heads=1, dropout=0.5):
        super(CrossRec, self).__init__()
        self.rna_idembedding = nn.Embedding(num_embeddings=in_channels_dict['rna_id'], embedding_dim=id_embedding_dims)
        self.atac_idembedding = nn.Embedding(num_embeddings=in_channels_dict['atac_id'], embedding_dim=id_embedding_dims)
        self.generation = generation
        self.save_attn_flag = save_attn_flag
        self.save_dir = save_dir
        # self.rna_feature_embedding = nn.Linear(in_channels_dict['rna_feature'], in_channels_dict['rna_feature_hidden'])
        # self.atac_feature_embedding = nn.Linear(in_channels_dict['atac_feature'], in_channels_dict['atac_feature_hidden'])
        self.cross_attention_layer = BiologicalDrivenGraphTransformer(
                    id_embedding_dims + in_channels_dict['rna_feature'],
                    id_embedding_dims + in_channels_dict['atac_feature'],
                    hidden_channels, num_heads=num_heads, dropout=dropout)

        self.rna_database = torch.load(rna_database_path)
        # Matching Head
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)  # RNA和ATAC的聚合输出
        self.fc2 = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.fc3 = nn.Linear(int(hidden_channels/2), out_channels)
        self.dropout = nn.Dropout(dropout)
        # Counter, for case study
        self.sample_counter = 0 

    # --------------------------------------------------------
    #  MATCHING
    # --------------------------------------------------------
    def forward_matching(self, data, device):
        x_dict = {k: v.float() for k, v in data.x_dict.items()}
        chrom_mask = data.chrom_mask

        # Saving non-zero index, for case study
        if (self.save_attn_flag) and (self.training==False):
            save_dir_atac = os.path.join(self.save_dir, "non_zero_atac_index")
            save_dir_rna = os.path.join(self.save_dir, "non_zero_rna_index")
            os.makedirs(save_dir_atac, exist_ok=True)
            os.makedirs(save_dir_rna, exist_ok=True)
            sample_id = self.sample_counter
            self.sample_counter += 1
            atac_non_zero_indices = data['atac'].non_zero_indices
            save_path = os.path.join(save_dir_atac, f"sample_{sample_id}.pt")
            torch.save(atac_non_zero_indices, save_path)
            rna_non_zero_indices = data['rna'].non_zero_indices
            save_path = os.path.join(save_dir_rna, f"sample_{sample_id}.pt")
            torch.save(rna_non_zero_indices, save_path)

        # ATAC encoding
        x_atac_id = self.atac_idembedding(x_dict['atac'][:,0].long())
        x_atac_feature = x_dict['atac'][:,1:]
        x_atac = torch.cat((x_atac_id, x_atac_feature), dim=1)

        # RNA encoding
        x_rna_id = self.rna_idembedding(x_dict['rna'][:,0].long())
        x_rna_feature = x_dict['rna'][:,1:]
        x_rna = torch.cat((x_rna_id, x_rna_feature), dim=1)

        # Biological-driven Graph Transformer
        x_rna, x_atac = self.cross_attention_layer(x_rna, x_atac, chrom_mask, save_attn_flag = self.save_attn_flag, save_dir = self.save_dir)

        # Graph-level Pooling
        x_rna_agg = global_mean_pool(x_rna, data["rna"].batch)
        x_atac_agg = global_mean_pool(x_atac, data["atac"].batch)
        x = torch.cat([x_rna_agg, x_atac_agg], dim=1)

        # Matching head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    # --------------------------------------------------------
    #  Translation
    # --------------------------------------------------------
    def forward_translation(self, data, device):
        x_dict = {k: v.float() for k, v in data.x_dict.items()}
        chrom_mask = data.chrom_mask

        # ATAC encoding
        x_atac_id = self.atac_idembedding(x_dict['atac'][:, 0].long())
        x_atac_feature = x_dict['atac'][:, 1:]
        x_atac = torch.cat((x_atac_id, x_atac_feature), dim=1)

        # Load RNA database (full set of RNA nodes)
        rna_db = torch.tensor(self.rna_database).long().to(device)
        x_rna_id = self.rna_idembedding(rna_db[:, 0])
        x_rna_feature = rna_db[:, 1:]
        ones = torch.ones(x_rna_feature.size(0), 1, device=device)
        x_rna = torch.cat((x_rna_id, ones, x_rna_feature), dim=1)

        # Crossmodal translation mode
        return self.cross_attention_layer(
            x_rna, x_atac,
            chrom_mask,
            generation=True
        )

    def forward_integration(self, data, device):
        x_dict = {k: v.float() for k, v in data.x_dict.items()}
        chrom_mask = data.chrom_mask

        # ATAC encoding
        x_atac_id = self.atac_idembedding(x_dict['atac'][:,0].long())
        x_atac_feature = x_dict['atac'][:,1:]
        x_atac = torch.cat((x_atac_id, x_atac_feature), dim=1)

        # RNA encoding
        x_rna_id = self.rna_idembedding(x_dict['rna'][:,0].long())
        x_rna_feature = x_dict['rna'][:,1:]
        x_rna = torch.cat((x_rna_id, x_rna_feature), dim=1)

        # Biological-driven Graph Transformer
        x_rna, x_atac = self.cross_attention_layer(x_rna, x_atac, chrom_mask)

        # Graph-level Pooling
        x_rna_agg = global_mean_pool(x_rna, data["rna"].batch)
        x_atac_agg = global_mean_pool(x_atac, data["atac"].batch)
        x = torch.cat([x_rna_agg, x_atac_agg], dim=1)
        return x
