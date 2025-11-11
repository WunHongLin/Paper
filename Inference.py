import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from Model import MLCL_Model
from tqdm import tqdm
from Train import MultiLabelDataset, precision_at_k
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, ndcg_score

if __name__ == "__main__":
    """
    推論參數
    """
    Parameters = {
        "config": "bert-base-uncased",
        "label_emb_path": "GCN_Embedding/label_GCN_AAPD_emb.npy",
        "num_labels": 54,
        "max_len": 512,
        "batch": 32,
        "device": "cuda",
        "model_path": "MLCL.pt",
        "dataset": "AAPD",
        "num_labels": 54,
        "tau": 0.05,
        "gamma": 0.6,
        "proto_momentum": 0.7
    }

    """
    載入模型並設置參數
    """
    print("Loading model...")
    model = MLCL_Model(Parameters["config"], 
                       Parameters["num_labels"], 
                       Parameters["label_emb_path"], 
                       Parameters['tau'], 
                       Parameters['proto_momentum']).to(Parameters['device'])
    model.load_state_dict(torch.load(Parameters["model_path"], map_location=Parameters["device"]))
    model.eval()

    test_df = pd.read_csv(f"{Parameters['dataset']}/test/test.csv")

    def encode_labels(label_str):
        labels = [int(x) for x in label_str.split()]
        vec = np.zeros(Parameters["num_labels"], dtype=np.float32)
        vec[labels] = 1.0
        return vec

    test_df["NumberLabels"] = test_df["NumberLabels"].apply(encode_labels)

    tokenizer = BertTokenizer.from_pretrained(Parameters["config"])
    test_dataset = MultiLabelDataset(test_df, tokenizer, Parameters["max_len"])
    test_loader = DataLoader(test_dataset, batch_size=Parameters["batch"])

    total_P1, total_P3, total_P5, total_nDCG3, total_nDCG5, hl, macro_f1, micro_f1, acc = 0, 0, 0, 0, 0, 0, 0, 0, 0
    step_count = 0

    print("start inference")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(Parameters['device'])
            attention_mask = batch["attention_mask"].to(Parameters['device'])
            labels = batch["labels"].to(Parameters['device'])

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

            total_P1 += P1
            total_P3 += P3
            total_P5 += P5
            total_nDCG3 += nDCG3
            total_nDCG5 += nDCG5

            preds_bin = (probs > 0.5).int().cpu().detach().numpy()
            micro_f1 += f1_score(labels_cpu, preds_bin, average='micro', zero_division=0)
            macro_f1 += f1_score(labels_cpu, preds_bin, average='macro', zero_division=0)
            hl += hamming_loss(labels_cpu, preds_bin)
            acc += sum(np.all(preds_bin == labels_cpu, axis=1))

            step_count += 1

        print("\n===== TEST RESULTS =====")
        print(f"P@1: {total_P1/step_count:.4f}")
        print(f"P@3: {total_P3/step_count:.4f}")
        print(f"P@5: {total_P5/step_count:.4f}")
        print(f"nDCG@3: {total_nDCG3/step_count:.4f}")
        print(f"nDCG@5: {total_nDCG5/step_count:.4f}")
        print(f"Micro-F1: {micro_f1/step_count:.4f}")
        print(f"Macro-F1: {macro_f1/step_count:.4f}")
        print(f"Hamming Loss: {hl/step_count:.4f}")
        print(f"Acc: {acc/len(test_df):.4f}")