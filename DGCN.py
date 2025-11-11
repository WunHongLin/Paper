# train_label_gcn_aapd.py
import math, random, os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

from Preprocess import DataPreprecessed
from FigureTool import Frequence, Coocurrence, LabelGraph

"""控制實驗隨機性"""
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

"""
將數值字串轉製成數字串列, 1 0 2 ===> [1, 0, 2]
"""
def parse_label_str(s):
    s = str(s).strip()
    return [int(x) for x in s.split()]

"""
應用PPMI, Label Occurence Frequency 建構初始化的A矩陣
"""
def build_label_stats(df, alpha_omega, num_labels=None, eps=1e-12):
    label_lists = df["NumberLabels"].apply(parse_label_str).tolist()

    """用來對應生成維度"""
    m = num_labels

    """初始化並計算標籤出現頻率以及其共現"""
    freq = np.zeros(m, dtype=np.int64)
    cooc = np.zeros((m, m), dtype=np.int64)

    for ls in label_lists:
        uniq = sorted(set(ls))
        for i in uniq:
            freq[i] += 1
        for i in range(len(uniq)):
            for j in range(i, len(uniq)):
                a, b = uniq[i], uniq[j]
                if a != b:
                    cooc[a, b] += 1
                    cooc[b, a] += 1

    """應用PPMI, 計算初始化的A0矩陣"""
    N = float(len(label_lists)) 
    p = (freq + eps) / (N + eps)
    # ω(f) = log(1+f)^α
    omega = np.power(np.log1p(freq.astype(np.float64)) + eps, alpha_omega)
    # p(ci,cj)
    pij = (cooc.astype(np.float64) + eps) / (N + eps)

    denom = np.outer(np.power(p, omega), np.power(p, omega))
    pmi = np.log((pij + eps) / (denom + eps))
    ppmi = np.maximum(pmi, 0.0)

    for i in range(m):
        ppmi[i, i] = 1.0

    """對稱正規畫"""
    d = np.clip(ppmi.sum(axis=1), a_min=eps, a_max=None)
    dinv_sqrt = np.power(d, -0.5)
    A_norm = ppmi * dinv_sqrt[:, None] * dinv_sqrt[None, :]

    A0 = torch.tensor(A_norm, dtype=torch.float32)

    return A0, freq, cooc, m

"""
產生初始化標籤向量
"""
@torch.no_grad()
def encode_label_texts(label_texts, bert_name="bert-base-uncased", max_len=32, device="cuda"):
    tok = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
    enc = AutoModel.from_pretrained(bert_name).to(device)
    enc.eval()

    batch = tok(label_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    out = enc(**batch)
    if getattr(out, "pooler_output", None) is not None:
        h = out.pooler_output          # [m, H]
    else:
        h = out.last_hidden_state[:, 0]  # [CLS]
    return h.detach()  # [m, H]

"""
GCN模型
"""
class LabelSemanticGCN(nn.Module):
    """
    - C : label embeddings(初始化自 BERT)
    - A0: 由 PPMI 取得的初始圖(固定 buffer)
    - 兩層 GCN(線性 + ReLU)，可選 Dropout
    - 訓練目標：讓 H H^T 的相似度近似 A0(以 MSE 損失）
    """
    def __init__(self, C0, A0, dropout=0.1, mix_beta=None):
        super().__init__()
        m, d0 = C0.shape
        self.m = m
        self.register_buffer("A0", A0)                      # [m,m], fixed
        self.C = nn.Parameter(C0.clone().detach())          # [m,d0], learnable

        d1 = d0
        self.gcn1 = nn.Linear(d0, d1)
        self.gcn2 = nn.Linear(d1, d1)
        self.dropout = nn.Dropout(dropout)
        self.mix_beta = mix_beta

    def forward(self):
        C = self.C                                          # [m, d]
        A_tilde = self.A0
        attn = torch.softmax(C @ C.t(), dim=-1)         # [m, m]
        A_tilde = self.mix_beta * self.A0 + (1.0 - self.mix_beta) * attn

        d = torch.clamp(A_tilde.sum(dim=1), min=1e-12)
        dinv_sqrt = torch.pow(d, -0.5)
        A_hat = A_tilde * dinv_sqrt[:, None] * dinv_sqrt[None, :]

        H = torch.relu(A_hat @ self.gcn1(C))
        H = self.dropout(H)
        H = A_hat @ self.gcn2(H)                            # [m, d1]
        return H
    
"""
重構損失
"""
def reconstruction_loss(H, A0):
    Hn = F.normalize(H, dim=1)
    S = torch.relu(Hn @ Hn.t())  # [m,m], in [0,1]
    return F.mse_loss(S, A0)


parameters = {
    "dataset": "AAPD",
    "num_class": 54,
    "config": "bert-base-uncased",
    "max_length": 32,
    "device": "cuda",
    "drop_out": 0.1,
    "mix_alpha": 5,
    "mix_beta": 0.5,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 100,
    "save_prefix": "label_GCN"
}

if __name__ == "__main__":
    if parameters["dataset"] == "AAPD": DataPreprecessed("AAPD")
    else: DataPreprecessed("EUR-Lex")

    df = pd.read_csv(f'./{parameters["dataset"]}/train/train.csv', sep=",").reset_index(drop=True)

    A0, freq, cooc, m = build_label_stats(df, num_labels=parameters["num_class"], alpha_omega=parameters['mix_alpha'])
    A0 = A0.to(parameters['device'])

    # Frequence(freq, parameters['dataset'])
    # Coocurrence(m, cooc, parameters['dataset'])
    # LabelGraph(A0, 30, 0.005, parameters['dataset'])

    label_texts = list(pd.read_json(f"./{parameters['dataset']}/label_to_index.json", type='series').to_dict().keys())
    C0 = encode_label_texts(label_texts, bert_name=parameters['config'], max_len=parameters['max_length'], device=parameters['device'])

    model = LabelSemanticGCN(C0=C0, A0=A0, dropout=parameters['drop_out'], mix_beta=parameters['mix_beta']).to(parameters['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])

    """
    開始訓練
    """
    print("Start Training...")

    best_loss, best_H = float("inf"), None
    patience = 0

    for epoch in range(1, parameters['epochs']+1):
        model.train()
        H = model()
        loss = reconstruction_loss(H, A0)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            cur = loss.item()
            if cur < best_loss:
                best_loss = cur
                best_H = H.detach().cpu().numpy()
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    print(f"Early stopped at epoch {epoch}")
                    break

        print(f"Epoch {epoch:03d} | loss={loss.item():.6f}")

    """
    儲存訓練資料
    """
    os.makedirs("GCN_Embedding", exist_ok=True)
    np.save(f"GCN_Embedding/{parameters['save_prefix']}_{parameters['dataset']}_emb.npy", best_H)
    torch.save(model.state_dict(), f"GCN_Embedding/{parameters['save_prefix']}_{parameters['dataset']}_model.pt")

    print("Finish Training...")