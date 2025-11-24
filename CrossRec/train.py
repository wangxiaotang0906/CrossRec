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

dataset_store_path = './interaction_graph_datasets/ISSAAC'
best_model_path = "./ISSAAC_test.pth"
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

def train(model, data, optimizer, device, alpha=1.0):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()

    # ------------------------------
    # 1. Matching Loss
    # ------------------------------
    match_output = model.forward_matching(data, device)
    match_loss = F.cross_entropy(match_output, data.y)

    # ------------------------------
    # 2. Generation Loss (only y=1)
    # ------------------------------
    pos_idx = (data.y == 1).nonzero(as_tuple=True)[0]
    if len(pos_idx) > 0:
        data_list = data.to_data_list()
        pos_list = [data_list[i] for i in pos_idx]
        pos_batch = Batch.from_data_list(pos_list).to(device)

        gen_pred = model.forward_translation(pos_batch, device)
        gen_loss = F.mse_loss(gen_pred, pos_batch['rna'].rna_expression.float().unsqueeze(0), reduction='mean')
    else:
        gen_loss = torch.tensor(0.0, device=device)

    # ------------------------------
    # 3. Total Loss with Adaptive alpha
    # ------------------------------
    # with torch.no_grad():
    #     if gen_loss.item() > 0:
    #         alpha = (0.1 * match_loss.item()) / gen_loss.item()
    #     else:
    #         alpha = 0.0
    loss = match_loss + alpha * gen_loss
    loss.backward()
    optimizer.step()

    return loss.item(), match_loss.item(), gen_loss.item()


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

def compute_foscttm(predicted: np.ndarray, true: np.ndarray) -> float:
    """
    Compute FOSCTTM between predicted and true RNA profiles.
    """
    N = predicted.shape[0]
    scores = []
    for i in range(N):
        d_true = np.linalg.norm(true[i] - predicted[i])
        d_all = np.linalg.norm(true[i] - predicted, axis=1)
        score = np.sum(d_all < d_true) / (N - 1)
        scores.append(score)
    return float(np.mean(scores))

def compute_per_cell_pearson(predicted: np.ndarray, true: np.ndarray) -> float:
    """
    Compute the average Pearson correlation across all cells.
    Args:
        predicted: numpy array of shape [num_cells, num_genes]
        true: numpy array of shape [num_cells, num_genes]
    Returns:
        average Pearson correlation coefficient across all cells
    """
    assert predicted.shape == true.shape
    num_cells = predicted.shape[0]
    correlations = []
    for i in range(num_cells):
        pred = predicted[i]
        gt = true[i]
        if np.std(pred) == 0 or np.std(gt) == 0:
            continue
        corr, _ = pearsonr(pred, gt)
        correlations.append(corr)
    if len(correlations) == 0:
        return 0.0
    return float(np.mean(correlations))

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
        # foscttm_score = compute_foscttm(all_preds, all_truths)
        # pearson = compute_per_cell_pearson(all_preds, all_truths)

        return rmse, mae

def main():
    set_seed(906)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_channels_dict = {'rna_id': 32208, 'atac_id': 169180, 'rna_feature': 34, 'atac_feature': 286, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #ISSAAC
    # in_channels_dict = {'rna_id': 29095, 'atac_id': 107194, 'rna_feature': 42, 'atac_feature': 294, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #10XPBMC
    # in_channels_dict = {'rna_id': 21478, 'atac_id': 340341, 'rna_feature': 36, 'atac_feature': 288, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #SHARE
    # in_channels_dict = {'rna_id': 28930, 'atac_id': 241757, 'rna_feature': 34, 'atac_feature': 286, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #SNARE
    model = CrossRec(in_channels_dict, id_embedding_dims=256, hidden_channels=128, out_channels=2, rna_database_path=rna_database_path, generation=False, layer_num=1, multihead=True, num_heads=1,
                             dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5) #128 128 0.3 5e-5 1e-6； 256 128 0.1 3e-5 1e-6
    print("model:", model)

    # Parameter quantity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameter: {total_params}")
    print(f"Trainable Parameter: {trainable_params}")

    # Loading dataset
    dataset = InteractionGraphDataset(root=dataset_store_path)
    print("dataset:", dataset)
    print("num_samples:", len(dataset))
    print("dataset samples:", dataset[0])
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                             [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    # For case study
    # test_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    train_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    best_MAE = 100
    best_score = -100
    # Resume training from checkpoint
    # model.load_state_dict(torch.load(best_model_path))
    # model.to(device)

    # Train
    for epoch in range(1, 100):
        epoch_loss = 0
        all_match_loss = 0
        all_gen_loss = 0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch', dynamic_ncols=True, leave=False)
        for data in pbar:
            data = data.to(device)
            loss, match_loss, gen_loss = train(model, data, optimizer, device, alpha=1.0)
            epoch_loss += loss
            all_match_loss += match_loss
            all_gen_loss += gen_loss
            pbar.set_postfix(loss=loss)
        pbar.close()

        avg_loss = epoch_loss / len(train_loader)
        avg_match_loss = all_match_loss / len(train_loader)
        avg_gen_loss = all_gen_loss / len(train_loader)
        epoch_time = time.time() - start_time

        # Val
        val_accuracy, val_P_Precision, val_N_Precision, val_P_Recall, val_N_Recall = matching_evaluate(model, val_loader, device)
        RMSE, MAE = translation_evaluate(model, val_loader, device)
        tqdm.write(f'Epoch {epoch}, Time: {epoch_time:.2f}s, Total loss: {avg_loss:.4f}, Matching loss: {avg_match_loss:.4f}, Translation loss: {avg_gen_loss:.4f}, '
                       f'Validation Accuracy: {val_accuracy:.4f}, '
                       f'Validation MAE: {MAE:.4f}, '
                   )

        # Saving better model
        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     torch.save(model.state_dict(), best_matching_model_path)
        #     print(f"Saved best model for matching of Epoch {epoch} with val_accuracy: {val_accuracy:.4f}")
        #
        # if MAE < best_MAE:
        #     best_MAE = MAE
        #     torch.save(model.state_dict(), best_translation_model_path)
        #     print(f"Saved best model for translation of Epoch {epoch} with MAE: {MAE:.4f}")

        if val_accuracy-MAE > best_score:
            best_score = val_accuracy-MAE
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model of Epoch {epoch} with val_accuracy: {val_accuracy:.4f} and MAE: {MAE:.4f}")


        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        time.sleep(1)

    # Painting loss and performance curve
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Loading best model
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    # Test
    test_accuracy, test_P_Precision, test_N_Precision, test_P_Recall, test_N_Recall = matching_evaluate(model, test_loader, device)
    RMSE,MAE = translation_evaluate(model, test_loader, device)
    print(f'Test Matching Accuracy: {test_accuracy:.4f}')
    print(f'Test P Precision: {test_P_Precision:.4f}')
    print(f'Test N Precision: {test_N_Precision:.4f}')
    # print(f'Test P Recall: {test_P_Recall:.4f}')
    # print(f'Test N Recall: {test_N_Recall:.4f}')
    print(f'Test Translation RMSE: {RMSE:.4f}')
    print(f'Test Translation MAE: {MAE:.4f}')
    # print(f"FOSCTTM score: {foscttm_score:.4f}")
    # print(f"Pearson score: {pearson:.4f}")


if __name__ == '__main__':
    main()