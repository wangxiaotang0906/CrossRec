import os
import json
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import itertools
import scipy.sparse
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

select_cell_type = False # Select partial cell types
selected_cell_type = [""] # Cell types for training should >=3.
for_discovery = False # If used for discovery, only positive samples are generated.
if_easy_neg = 1
neg_easy_ratio = 1

dataset_store_path = '../interaction_graph_datasets/SHARE_test'

rna_anndata = sc.read_h5ad("../original_datasets/SHARE-seq/Ma-2020-RNA.h5ad")
print('rna_anndata:',rna_anndata) #32231 cells, 21478 genes
atac_anndata = sc.read_h5ad("../original_datasets/SHARE-seq/Ma-2020-ATAC.h5ad")
print('atac_anndata:',atac_anndata) #32231 cells, 340341 peaks

if scipy.sparse.issparse(rna_anndata.X):
    rna_anndata.X = rna_anndata.X.toarray()
    atac_anndata.X = atac_anndata.X.toarray()
else:
    rna_anndata.X = np.array(rna_anndata.X)
    atac_anndata.X = np.array(atac_anndata.X)

if select_cell_type==True:
    rna_anndata = rna_anndata[rna_anndata.obs['cell_type'].isin(selected_cell_type)].copy()
    atac_anndata = atac_anndata[atac_anndata.obs['cell_type'].isin(selected_cell_type)].copy()

total_cells = len(rna_anndata)
if total_cells!=len(atac_anndata):
    print("ERROR: different size of rna and atac sequence!")
    exit()

rna_var_df = rna_anndata.var.copy()[['means','variances','variances_norm','chrom','strand','highly_variable','highly_variable_rank']]
rna_var_df['original_index'] = rna_var_df.index
atac_var_df = atac_anndata.var.copy()[['chrom']]
atac_var_df['original_index'] = atac_var_df.index

# processing index
rna_var_df = rna_var_df.reset_index(drop=True)
# print(rna_var_df)
atac_var_df = atac_var_df.reset_index(drop=True)
# print(atac_var_df)

# processing chrom (one-hot)
all_categories = pd.concat([rna_var_df['chrom'], atac_var_df['chrom']]).unique()
rna_var_df['chrom'] = pd.Categorical(rna_var_df['chrom'], categories=all_categories).codes
atac_var_df['chrom'] = pd.Categorical(atac_var_df['chrom'], categories=all_categories).codes
encoder = OneHotEncoder(categories=[range(len(all_categories))], sparse_output=False)

# processing strand, highly_variable
mapping = {'-': 0, '+': 1}
rna_var_df['strand'] =  rna_var_df['strand'].map(mapping)
rna_var_df['highly_variable'] = rna_var_df['highly_variable'].astype(int)

# df to numpy, index as unique id
rna_var_rowindex = rna_var_df.index.values
rna_var_rowindex = rna_var_rowindex.astype(int)
rna_var_chrom_np=rna_var_df[['chrom']].to_numpy().reshape(-1, 1)
rna_var_chrom_np_onehot = encoder.fit_transform(rna_var_chrom_np)
rna_var_np=rna_var_df[['means','variances_norm','strand','highly_variable']].to_numpy()
rna_var_np = np.hstack((rna_var_rowindex.reshape(-1,1),rna_var_chrom_np_onehot,rna_var_np))
print("rna row features shape:",rna_var_np.shape)
print("rna row features:",rna_var_np)

# RNA database
os.makedirs(dataset_store_path, exist_ok=True)
torch.save(rna_var_np, os.path.join(dataset_store_path, f"rna_database.pt"))

# processing ATAC_DNA embedding
def dna_sequence_to_numeric(sequence):
    base_mapping = {
        'A': 0.25,
        'T': 0.5,
        'G': 0.75,
        'C': 1.0,
        'N': 0.0
    }
    return [base_mapping.get(base, 0.0) for base in sequence]
descriptions = []
sequences = []
max_len = 256
with open("../original_datasets/SHARE-seq/SHARE_peak_output.fa", 'r') as f:
    seq = ""
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if seq:
                seq = seq[:max_len] + 'N' * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
                sequences.append(seq)
                seq = ""
            descriptions.append(line[1:])
        else:
            seq += line
    if seq:
        seq = seq[:max_len] + 'N' * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
        sequences.append(seq)

atac_dna_df = pd.DataFrame({
        "Description": descriptions,
        "Sequence": sequences
    })
atac_var_dna_df = pd.concat([atac_var_df.reset_index(drop=True), atac_dna_df[['Sequence']].reset_index(drop=True)], axis = 1)
atac_var_dna_df['Sequence'] = atac_var_dna_df['Sequence'].fillna('N' * 256)
atac_var_dna_np= atac_var_dna_df[['Sequence']].to_numpy().reshape(-1, 1)
for i in range(atac_var_dna_np.shape[0]):
    sequence = atac_var_dna_np[i, 0]
    numeric_sequence = dna_sequence_to_numeric(sequence)
    atac_var_dna_np[i, 0] = numeric_sequence
atac_var_dna_np = np.array(atac_var_dna_np.tolist(), dtype=np.float32)
atac_var_dna_np = np.squeeze(atac_var_dna_np, axis=1)

# df to numpy, index as unique id
atac_var_rowindex = atac_var_df.index.values
atac_var_rowindex = atac_var_rowindex.astype(int)
atac_var_chrom_np=atac_var_df[['chrom']].to_numpy().reshape(-1, 1)
atac_var_chrom_np_onehot = encoder.fit_transform(atac_var_chrom_np)
atac_var_np = np.hstack((atac_var_rowindex.reshape(-1,1),atac_var_chrom_np_onehot,atac_var_dna_np))
print("atac row features shape:",atac_var_np.shape)
print("atac row features:",atac_var_np)

# ATAC database
os.makedirs(dataset_store_path, exist_ok=True)
torch.save(atac_var_np, os.path.join(dataset_store_path, f"atac_database.pt"))

# Information of cell types
all_cell_df = rna_anndata.obs.copy()[['cell_type']]
all_cell_df = all_cell_df.reset_index(drop=True)
print("cell type distribution:", rna_anndata.obs['cell_type'].value_counts())
grouped = all_cell_df.groupby('cell_type')
cell_type_cat = rna_anndata.obs['cell_type'].astype('category')
cell_type_codes_np = cell_type_cat.cat.codes.to_numpy()
cell_type_names = list(cell_type_cat.cat.categories)
os.makedirs(dataset_store_path, exist_ok=True)
with open(os.path.join(dataset_store_path, "cell_type_vocab.json"), "w", encoding="utf-8") as f:
    json.dump({"idx2name": cell_type_names}, f, ensure_ascii=False, indent=2)

# easy_neg
if if_easy_neg == 1 and for_discovery == False:
    all_cells = all_cell_df.index.tolist()
    all_types = all_cell_df['cell_type'].tolist()
    total_easy_negative_pairs = total_cells * neg_easy_ratio
    easy_negative_pairs = []

    while len(easy_negative_pairs) < total_easy_negative_pairs:
        cell1, cell2 = np.random.choice(all_cells, size=2, replace=False)
        if all_cell_df.loc[cell1, 'cell_type'] != all_cell_df.loc[cell2, 'cell_type']:
            easy_negative_pairs.append((cell1, cell2))

# hard_neg
if if_easy_neg == 0 and for_discovery == False:
    total_hard_negative_pairs = total_cells * neg_easy_ratio
    hard_negative_pairs = []
    cell_type_proportions = grouped.size() / total_cells
    cell_type_negative_counts = (cell_type_proportions * total_hard_negative_pairs).astype(int)
    for cell_type, group in grouped:
        cells = group.index.tolist()
        negative_count = cell_type_negative_counts[cell_type]
        if len(cells) > 1 and negative_count > 0:
            pairs = list(itertools.combinations(cells, 2))
            if len(pairs) > negative_count:
                selected_indices = np.random.choice(len(pairs), size=negative_count, replace=False)
                selected_pairs = [pairs[i] for i in selected_indices]
            hard_negative_pairs.extend(selected_pairs)


# create interaction graph with ATAC and RNA from a cell (pos), or from two different cells (neg)
def create_interaction_graph(cellnumber_0, cellnumber_1=None, if_pos=1):
    data = HeteroData()
    rna_non_zero_indices = rna_anndata.X[cellnumber_0].nonzero()[0]
    rna_chorm_type = rna_var_chrom_np[rna_non_zero_indices]
    rna_x_expression = rna_anndata.X[cellnumber_0][rna_non_zero_indices]
    rna_x = np.hstack((rna_var_np[rna_non_zero_indices, 0].reshape(-1, 1), rna_x_expression.reshape(-1, 1),
                       rna_var_np[rna_non_zero_indices, 1:]))
    data['rna'].x = torch.tensor(rna_x)
    # rna_expression generation
    data['rna'].rna_expression = torch.tensor(rna_anndata.X[cellnumber_0])
    data['rna'].non_zero_indices = torch.tensor(rna_non_zero_indices)

    if if_pos == 1:
        cellnumber_1 = cellnumber_0

    atac_non_zero_indices = atac_anndata.X[cellnumber_1].nonzero()[0]
    atac_chorm_type = atac_var_chrom_np[atac_non_zero_indices]
    atac_x_expression = atac_anndata.X[cellnumber_1][atac_non_zero_indices]
    atac_x = np.hstack((atac_var_np[atac_non_zero_indices, 0].reshape(-1, 1), atac_x_expression.reshape(-1, 1),
                        atac_var_np[atac_non_zero_indices, 1:]))
    data['atac'].x = torch.tensor(atac_x)
    # rna_expression generation
    data['atac'].atac_expression = torch.tensor(atac_anndata.X[cellnumber_1])
    data['atac'].non_zero_indices = torch.tensor(atac_non_zero_indices)

    # prior edges
    chrom_mask = (rna_chorm_type[:, None] == atac_chorm_type[None, :])
    data.chrom_mask = torch.tensor(chrom_mask, dtype=torch.bool)

    return data


class InteractionGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, if_easy_neg=1):
        self.if_easy_neg = if_easy_neg
        super(InteractionGraphDataset, self).__init__(root, transform, pre_transform)
        os.makedirs(self.processed_dir, exist_ok=True)
        if not os.path.exists(self.processed_paths[0]):
            print(f"Processed data not found. Generating and saving dataset...")
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'graph_{i}.pt' for i in range(total_cells * 2)]

    def download(self):
        pass

    def process(self):
        for i in tqdm(range(total_cells), desc="Processing Positive Examples"):
            data = create_interaction_graph(i)
            data.y = torch.tensor(1)
            data.cell_type = torch.tensor([int(cell_type_codes_np[i])], dtype=torch.long)
            torch.save(data, os.path.join(self.processed_dir, f'graph_{i}.pt'))
        print('Positive examples done.')

        if for_discovery == False:
            j = total_cells
            if self.if_easy_neg == 1:
                for m, n in tqdm(easy_negative_pairs, desc="Processing Easy Negative Pairs"):
                    data = create_interaction_graph(m, n, 0)
                    data.y = torch.tensor(0)
                    data.cell_type_id = torch.tensor([-1], dtype=torch.long)
                    torch.save(data, os.path.join(self.processed_dir, f'graph_{j}.pt'))
                    j = j + 1
            else:
                for m, n in tqdm(hard_negative_pairs, desc="Processing Hard Negative Pairs"):
                    data = create_interaction_graph(m, n, 0)
                    data.y = torch.tensor(0)
                    data.cell_type_id = torch.tensor([-1], dtype=torch.long)
                    torch.save(data, os.path.join(self.processed_dir, f'graph_{j}.pt'))
                    j = j + 1
            print('Negative examples done.')

    def __getitem__(self, idx):
        file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(file_path)
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.processed_file_names)


dataset = InteractionGraphDataset(root=dataset_store_path, if_easy_neg=if_easy_neg)

