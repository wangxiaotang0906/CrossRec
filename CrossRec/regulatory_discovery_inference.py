import time
import random
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader, HeteroData
from torch.utils.data import random_split
from model import CrossRec
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from torch_geometric.data import Batch

dataset_store_path = './interaction_graph_datasets/ISSAAC_discover'
best_model_path = "./ISSAAC_test.pth"
save_path = "./ISSAAC_test_attention" # Storage path for the attention matrix used for discovery
rna_database_path = os.path.join(dataset_store_path, f"rna_database.pt")
atac_database_path = os.path.join(dataset_store_path, f"atac_database.pt")

class InteractionGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, if_easy_neg=1):
        self.if_easy_neg = if_easy_neg
        super(InteractionGraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset_length = len([f for f in os.listdir(self.processed_dir)
                                   if f.startswith('graph_') and f.endswith('.pt')])
        if self.dataset_length == 0:
            print("No processed 'graph_*.pt' files found. Please ensure the dataset is generated correctly.")
            raise FileNotFoundError("No processed data files found in the processed directory.")
        if not os.path.exists(self.processed_paths[0]):
            print("Processed data not found. Please ensure the dataset is generated correctly.")
            raise FileNotFoundError("Required processed data file is missing.")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'graph_{i}.pt' for i in range(self.dataset_length)]

    def download(self):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        file_path = self.processed_paths[idx]
        data = torch.load(file_path, weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.dataset_length

def set_seed(seed=906):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # Numpy random seed
    torch.manual_seed(seed)  # PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch seed for GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable cudnn's auto-tuner

def matching_evaluate(model, loader, device):
    model.eval()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for data in loader:
        data = data.to(device)
        out = model.forward_matching(data, device)
        pred = out.argmax(dim=1)
        # print(f"label:{data.y},pred:{pred}")
        TP += ((pred == 1) & (data.y == 1)).sum().item()
        FP += ((pred == 1) & (data.y == 0)).sum().item()
        TN += ((pred == 0) & (data.y == 0)).sum().item()
        FN += ((pred == 0) & (data.y == 1)).sum().item()
    if (TP+TN+FP+FN)==0:
        Accuracy = 0
    else:
        Accuracy = (TP+TN) / (TP+TN+FP+FN)

    if (TP+FP) == 0:
        P_Precision = np.nan #'all_predict_N'
    else:
        P_Precision = TP / (TP+FP)

    if (TN+FN) == 0:
        N_Precision = np.nan #'all_predict_P'
    else:
        N_Precision = TN / (TN+FN)

    if (TP+FN) == 0:
        P_Recall = 0
    else:
        P_Recall= TP / (TP+FN)

    if (FP+TN) == 0:
        N_Recall = 0
    else:
        N_Recall= TN / (FP+TN)

    return Accuracy, P_Precision, N_Precision, P_Recall, N_Recall

def translation_evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if data.y == 1:
                pred_rna = model.forward_translation(data, device).squeeze(0)
                truth_rna = data['rna'].rna_expression
                truth_rna_binary = (truth_rna > 0).float()
                all_preds.append(pred_rna.cpu().numpy())
                all_truths.append(truth_rna_binary.cpu().numpy())

        all_preds = np.stack(all_preds, axis=0)
        all_truths = np.stack(all_truths, axis=0)

        # Evaluation Metrics
        mse = mean_squared_error(all_truths, all_preds)
        mae = mean_absolute_error(all_truths, all_preds)
        rmse = np.sqrt(mse)

        return rmse, mae

def main():
    set_seed(906)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels_dict = {'rna_id': 32208, 'atac_id': 169180, 'rna_feature': 34, 'atac_feature': 286, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #ISSAAC
    # in_channels_dict = {'rna_id': 29095, 'atac_id': 107194, 'rna_feature': 42, 'atac_feature': 38, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #10XPBMC
    # in_channels_dict = {'rna_id': 21478, 'atac_id': 340341, 'rna_feature': 36, 'atac_feature': 32, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #SHARE
    # in_channels_dict = {'rna_id': 28930, 'atac_id': 241757, 'rna_feature': 34, 'atac_feature': 30, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #SNARE

    model = CrossRec(in_channels_dict, id_embedding_dims=256, hidden_channels=128, out_channels=2, rna_database_path=rna_database_path, generation=False, layer_num=1, multihead=True, num_heads=1,
                             dropout=0.1).to(device)
    print("model:", model)

    # Parameter quantity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameter: {total_params}")
    print(f"Trainable Parameter: {trainable_params}")

    # Loading case study dataset
    dataset = InteractionGraphDataset(root=dataset_store_path)
    print("dataset:", dataset)
    print("num_samples:", len(dataset))
    print("dataset samples:", dataset[0])
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    model.save_attn_flag = True
    model.save_dir = save_path
    # Test
    test_accuracy, test_P_Precision, test_N_Precision, test_P_Recall, test_N_Recall = matching_evaluate(model, test_loader, device)
    RMSE,MAE = translation_evaluate(model, test_loader, device)
    print(f'Test Matching Accuracy: {test_accuracy:.4f}')
    print(f'Test P Precision: {test_P_Precision:.4f}')
    print(f'Test N Precision: {test_N_Precision:.4f}')
    print(f'Test P Recall: {test_P_Recall:.4f}')
    print(f'Test N Recall: {test_N_Recall:.4f}')
    print(f'Test Translation RMSE: {RMSE:.4f}')
    print(f'Test Translation MAE: {MAE:.4f}')

if __name__ == '__main__':
    main()