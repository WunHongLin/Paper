import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np

"""
模型架構
"""
class MLCL_Model(nn.Module):
    def __init__(self, bert_name, num_labels, label_emb_path, tau, proto_momentum, proj_dim=256):
        super(MLCL_Model, self).__init__()
        self.num_labels = num_labels
        self.proj_dim = proj_dim
        """
        預訓練模型
        """
        self.bert = BertModel.from_pretrained(bert_name)
        hidden_size = self.bert.config.hidden_size

        """
        載入label representaion
        """
        # [num_labels, 768]
        label_emb = np.load(label_emb_path)
        self.label_emb = nn.Parameter(torch.tensor(label_emb, dtype=torch.float32), requires_grad=False)

        """
        進行注意力機制
        """
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        """
        分類階段
        """
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
        """
        投射到向量空間，以進行對比學習
        """
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim) #如果要改的話就把*NUM_LABELS拿掉 *num_labels
        )

        """
        prototype緩慢更新策略
        """
        self.register_buffer("prototypes", torch.zeros(num_labels, proj_dim))
        self.proto_momentum = proto_momentum
        self.tau = tau
        self.eps = 1e-8

    @torch.no_grad()
    def ema_update_prototypes(self, z, labels, label_contrib):
        """
        計算各自在batch中出現幾次, z ===> [B, L, D(768)]
        """
        pos_mask = (labels > 0).float()
        counts = pos_mask.sum(dim=0)

        """
        將每一個投射向量乘上對應的標籤貢獻度，並計算該batch的prototype
        """
        weighted = z * label_contrib.unsqueeze(-1) 
        summed = (weighted * pos_mask.unsqueeze(-1)).sum(dim=0)

        upd_idx = torch.where(counts > 0)[0]
        if upd_idx.numel() == 0:
            return
        
        avg = summed[upd_idx] / counts[upd_idx].unsqueeze(-1).clamp_min(self.eps)

        """
        緩慢更新
        """
        if labels is not None:
            self.prototypes[upd_idx] = ((1.0 - self.proto_momentum) * self.prototypes[upd_idx] + self.proto_momentum * avg.detach())

    def compute_contrastive_loss(self, z, labels, label_contrib):
        """
        取出投射向量維度
        """
        B, L, D = z.shape
        P = F.normalize(self.prototypes, dim=-1)          # [L, D]
        loss_batch = []

        """
        開始計算loss
        """
        for i in range(B):
            z_i = z[i]                              
            y_i = labels[i].float()                  
            d_i = label_contrib[i]                         
            pos_mask = (y_i > 0)

            """
            計算每一個prototype跟向量的相似度，形成二微陣列
            """
            S = torch.matmul(z_i, P.t())
            logits = S / self.tau
            logsumexp = torch.logsumexp(logits, dim=1)

            diag = torch.diagonal(logits)                  
            
            """
            計算公式loss
            """
            per_label_loss = -(diag - logsumexp) * d_i
            loss_i = per_label_loss[pos_mask].sum() / pos_mask.sum().clamp_min(self.eps)
            loss_batch.append(loss_i)

        return torch.stack(loss_batch).mean()

    def forward(self, input_ids, attention_mask, labels):
        """
        取的文章特徵表示
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # [batch, seq_len, hidden]

        """
        將文章與標籤進行attention
        """
        batch_size = H.size(0)
        label_emb_exp = self.label_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_labels, hidden]
        attn_output, attn_weights = self.attention(query=label_emb_exp, key=H, value=H)
        weighted_H = torch.bmm(attn_weights, H)
        label_scores = torch.softmax(weighted_H.norm(dim=-1), dim=-1)
        print(label_scores)

        """
        讓H以及L進行attention， 得到加權之後的文本表示，並以此取出標籤貢獻度
        """

        """
        取得加權後的文本特徵表示 111
        """
        attn_probs = torch.softmax(attn_weights, dim=-1)  # [batch, num_labels, seq_len]
        label_context = torch.bmm(attn_probs, H)  # [batch, num_labels, hidden]
        pooled = label_context.mean(dim=1)  # [batch, hidden]

        """
        這裡要根據attn_probs去產生對應的文章-標籤貢獻分布 111
        """
        token_strength = H.norm(dim=-1)  # [batch, seq_len]
        label_importance = (attn_probs * token_strength.unsqueeze(1)).sum(dim=-1)  # [batch, num_labels]
        mask = (label_importance > 0.0).float() 
        label_importance = label_importance * mask
        label_contrib = label_importance / (label_importance.sum(dim=-1, keepdim=True) + 1e-9)

        """
        進行分類
        """
        logits = self.classifier(attn_output)
        logits = torch.einsum('bld,ld->bl', attn_output, self.classifier.weight) + self.classifier.bias  # [B, L]
        probs = self.sigmoid(logits)

        """
        呼叫投射網路並動態更新PROTOTYPE
        """
        """
        H版本
        """
        # doc_emb = H.mean(dim=1)
        # z = self.projection(doc_emb).view(-1, self.num_labels, self.proj_dim)
        # z = F.normalize(z, dim=-1)
        """
        label_context版本
        """
        z = self.projection(attn_output)
        z = F.normalize(z, dim=-1)

        if labels is not None:
            self.ema_update_prototypes(z, labels, label_contrib)

        return logits, probs, attn_probs, z, label_contrib