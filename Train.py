import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from Model import MLCL_Model
from sklearn.metrics import ndcg_score
import time

"""
資料封裝
"""
class MultiLabelDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["Texts"].tolist()
        self.labels = df["NumberLabels"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

"""
指標函式
"""  
def precision_at_k(preds, target, k):
    batch_size = preds.size(0)
    precisions = []
    for i in range(batch_size):
        topk_idx = torch.topk(preds[i], k).indices
        true_labels = torch.where(target[i]==1)[0]
        correct = len(set(topk_idx.tolist()) & set(true_labels.tolist()))
        precisions.append(correct / k)

    return sum(precisions) / batch_size

"""
訓練函式
"""
def train_model(model, dataloader, optimizer, criterion, device, gamma):
    model.train()

    train_loss, train_P1, train_P3, train_P5, train_nDCG3, train_nDCG5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, probs, attn_probs, z, label_contrib = model(input_ids, attention_mask, labels)

        """
        計算相關指標
        """
        P1 = precision_at_k(probs, labels, k=1)
        P3 = precision_at_k(probs, labels, k=3)
        P5 = precision_at_k(probs, labels, k=5)
        logits_cpu = probs.cpu().detach().numpy()
        labels_cpu = labels.cpu().detach().numpy()
        nDCG3 = ndcg_score(labels_cpu, logits_cpu, k=3)
        nDCG5 = ndcg_score(labels_cpu, logits_cpu, k=5)

        """
        各自推倒bce_loss 以及cl_loss
        """
        bce_loss = criterion(logits, labels)
        cl_loss = model.compute_contrastive_loss(z, labels, label_contrib)
        loss = bce_loss + gamma * cl_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_P1 += P1
        train_P3 += P3
        train_P5 += P5
        train_nDCG3 += nDCG3
        train_nDCG5 += nDCG5

    return {
        "loss": train_loss / len(dataloader),
        "p1": train_P1/ len(dataloader),
        "p3": train_P3/ len(dataloader),
        "p5": train_P5/ len(dataloader),
        "ndcg3": train_nDCG3/ len(dataloader),
        "ndcg5": train_nDCG5/ len(dataloader)
    }

"""
評估函式
"""
def evaluate(model, dataloader, criterion, device, gamma):
    model.eval()

    val_loss, val_P1, val_P3, val_P5, val_nDCG3, val_nDCG5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            """
            評估時不需要更新prototype
            """
            logits, probs, attn_probs, z, label_contrib = model(input_ids, attention_mask, labels=None)

            """
            計算相關指標
            """
            P1 = precision_at_k(probs, labels, k=1)
            P3 = precision_at_k(probs, labels, k=3)
            P5 = precision_at_k(probs, labels, k=5)
            logits_cpu = probs.cpu().detach().numpy()
            labels_cpu = labels.cpu().detach().numpy()
            nDCG3 = ndcg_score(labels_cpu, logits_cpu, k=3)
            nDCG5 = ndcg_score(labels_cpu, logits_cpu, k=5)

            bce_loss = criterion(logits, labels)
            cl_loss = model.compute_contrastive_loss(z, labels, label_contrib)
            loss = bce_loss + gamma * cl_loss

            val_loss += loss.item()
            val_P1 += P1
            val_P3 += P3
            val_P5 += P5
            val_nDCG3 += nDCG3
            val_nDCG5 += nDCG5

        return {
            "loss": val_loss / len(dataloader),
            "p1": val_P1/ len(dataloader),
            "p3": val_P3/ len(dataloader),
            "p5": val_P5/ len(dataloader),
            "ndcg3": val_nDCG3/ len(dataloader),
            "ndcg5": val_nDCG5/ len(dataloader)
        }

"""
early stop
"""
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, path="MLCL.pt"):
        """
        patience: 允許連續多少 epoch 沒改善
        delta: 判定改善的最小幅度
        path: 儲存最佳模型的路徑
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == "__main__":
    """
    訓練相關參數
    """
    Parameters = {
        "batch": 1,
        "epoch": 1,
        "config": "bert-base-uncased",
        "label_emb_path": "GCN_Embedding/label_GCN_AAPD_emb.npy",
        "lr": 2e-5,
        "max_len": 512,
        "device": "cuda",
        "dataset": "AAPD",
        "num_labels": 54,
        "tau": 0.05,
        "gamma": 0.6,
        "proto_momentum": 0.7
    }

    """
    讀取並轉換標籤資訊成multi-hot格式
    """
    train_df = pd.read_csv(f"{Parameters['dataset']}/train/train.csv").sample(1)
    val_df = pd.read_csv(f"{Parameters['dataset']}/val/val.csv").sample(1)

    def encode_labels(label_str):
        labels = [int(x) for x in label_str.split()]
        vec = np.zeros(Parameters["num_labels"], dtype=np.float32)
        vec[labels] = 1.0
        return vec
    
    train_df["NumberLabels"] = train_df["NumberLabels"].apply(encode_labels)
    val_df["NumberLabels"] = val_df["NumberLabels"].apply(encode_labels)

    """
    設定tokenizer以及將資料封裝
    """
    tokenizer = BertTokenizer.from_pretrained(Parameters["config"])
    train_dataset = MultiLabelDataset(train_df, tokenizer, Parameters["max_len"])
    val_dataset = MultiLabelDataset(val_df, tokenizer, Parameters["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=Parameters["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Parameters["batch"])

    """
    模型相關設定
    """
    model = MLCL_Model(Parameters["config"], 
                       Parameters["num_labels"], 
                       Parameters["label_emb_path"], 
                       Parameters['tau'], 
                       Parameters['proto_momentum']).to(Parameters['device'])
    optimizer = optim.AdamW(model.parameters(), lr=Parameters["lr"])
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=3, delta=1e-4, path="MLCL.pt")

    """
    開始訓練
    """
    print("Start Training...")

    for epoch in range(Parameters["epoch"]):
        st_time = time.time()

        print(f"\n===== Epoch {epoch+1} / {Parameters['epoch']} =====")
        train_result = train_model(model, train_loader, optimizer, criterion, Parameters["device"], Parameters['gamma'])
        val_result = evaluate(model, val_loader, criterion, Parameters["device"], Parameters['gamma'])

        print('[epoch %d] cost time: %.4f s'%(epoch + 1, time.time() - st_time))
        print('         loss     p@1     p@3     p@5   nDCG@3  nDCG@5')
        print('train | %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'%(train_result['loss'], train_result['p1'], train_result['p3'], train_result['p5'], train_result['ndcg3'], train_result['ndcg5']))
        print('val   | %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n'%(val_result['loss'], val_result['p1'], val_result['p3'], val_result['p5'], val_result['ndcg3'], val_result['ndcg5']))

        early_stopper(val_result["loss"], model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Finish Training...")
            break
        # torch.save(model.state_dict(), "MLCL.pt")