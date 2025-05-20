import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from conformer.conformer.model_def import Conformer

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossAttention(nn.Module):
    """交叉注意力模块 - 一层交叉注意力加一层自注意力"""
    def __init__(self, query_dim, key_dim, heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(query_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )
        
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(query_dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(query_dim)
        self.norm4 = nn.LayerNorm(query_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )
        
        # 投影层
        self.proj_key = nn.Linear(key_dim, query_dim)
        self.proj_value = nn.Linear(key_dim, query_dim)
        
    def forward(self, query, key, value, key_padding_mask=None):
        # 有音频输入时，先进行交叉注意力
        # 投影key和value到query的维度
        key_proj = self.proj_key(key)
        value_proj = self.proj_value(value)
        
        # 交叉注意力
        query_norm = self.norm1(query)
        cross_attn_output, _ = self.cross_attn(query_norm, key_proj, value_proj, 
                                             key_padding_mask=key_padding_mask)
        query = query + cross_attn_output
        query = query + self.ffn1(self.norm2(query))
        
        # 然后进行自注意力
        query_norm = self.norm3(query)
        self_attn_output, _ = self.self_attn(query_norm, query_norm, query_norm)
        query = query + self_attn_output
        query = query + self.ffn2(self.norm4(query))
        
        return query
    

class MMKWS2(nn.Module):
    def __init__(
        self,
        # anchor
        text_dim=64,
        audio_dim=1024,
        hidden_dim=128,
        # compare
        dim=80,
        encoder_dim=128,
        num_encoder_layers=6,
        num_attention_heads=4,
        dropout=0.1,
        num_transformer_layers=2
    ):
        super(MMKWS2, self).__init__()
        # 音频嵌入降维
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        # 文本嵌入投影
        self.text_proj = nn.Embedding(num_embeddings=402, embedding_dim=hidden_dim) # 401 + padding -1
        # 位置编码
        self.pos_enc = PositionalEncoding(hidden_dim)
        # 交叉注意力模块
        self.cross_attn = CrossAttention(hidden_dim, hidden_dim, heads=num_attention_heads, dropout=dropout)
        
        # Conformer层
        self.conformer = Conformer(
            input_dim=dim,
            encoder_dim=encoder_dim,
            num_encoder_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
        )
        
        # 特征映射层（将conformer输出维度映射到hidden_dim）
        self.feat_proj = nn.Linear(encoder_dim, hidden_dim)
        
        # Transformer层
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        
        # GRU分类器
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 序列标签预测
        self.seq_classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, anchor_wave_embedding, anchor_text_embedding, compare_wave, compare_lengths):
        batch_size = anchor_wave_embedding.size(0)
        
        # 1. 处理anchor_text嵌入
        text_feat = self.text_proj(anchor_text_embedding)  # [B, S, hidden_dim]
        text_feat = self.pos_enc(text_feat)
        
        # 2. 处理anchor_wave音频嵌入
        audio_feat = self.audio_proj(anchor_wave_embedding)  # [B, S, hidden_dim]
        audio_feat = self.pos_enc(audio_feat)
        
        # 3. 交叉注意力：文本和音频特征融合
        fused_feat = self.cross_attn(text_feat, audio_feat, audio_feat, key_padding_mask=None)  # [B, S, hidden_dim]
        
        # 4. 处理compare_wave的fbank特征
        compare_feat = self.conformer(compare_wave, compare_lengths)[0]  # [B, T, encoder_dim]        
        compare_feat = self.feat_proj(compare_feat)  # [B, T, hidden_dim]
        compare_feat = self.pos_enc(compare_feat)

        # 5. 合并特征
        text_len = fused_feat.size(1)
        combined_feat = torch.cat([fused_feat, compare_feat], dim=1)  # [B, S+T, hidden_dim]
        combined_feat = self.transformer_encoder(combined_feat) # [B, S+T, hidden_dim]
            
        # 7. GRU分类
        gru_out, _ = self.gru(combined_feat)  # [B, S+T, hidden_dim*2]
        
        # 全局分类
        global_feat = gru_out[:, -1, :]  # 取最后一个时间步
        logits = self.classifier(global_feat).squeeze(-1)  # [B]
        
        # 序列标签预测
        seq_logits = self.seq_classifier(combined_feat[:, :text_len, :]).squeeze(-1)  # [B, S]
        return logits, seq_logits
    
    
    def enrollment(self, anchor_wave_embedding, anchor_text_embedding):
        batch_size = anchor_wave_embedding.size(0)
        
        # 1. 处理anchor_text嵌入
        text_feat = self.text_proj(anchor_text_embedding)  # [B, S, hidden_dim]
        text_feat = self.pos_enc(text_feat)
        
        # 2. 处理anchor_wave音频嵌入
        audio_feat = self.audio_proj(anchor_wave_embedding)  # [B, S, hidden_dim]
        audio_feat = self.pos_enc(audio_feat)
        
        # 3. 交叉注意力：文本和音频特征融合
        fused_feat = self.cross_attn(text_feat, audio_feat, audio_feat, key_padding_mask=None)  # [B, S, hidden_dim]

        return fused_feat
    
    def verification(self, fused_feat, compare_wave, compare_lengths):
        batch_size = fused_feat.size(0)

        # 4. 处理compare_wave的fbank特征
        compare_feat = self.conformer(compare_wave, compare_lengths)[0]  # [B, T, encoder_dim]        
        compare_feat = self.feat_proj(compare_feat)  # [B, T, hidden_dim]
        compare_feat = self.pos_enc(compare_feat)

        # 5. 合并特征
        text_len = fused_feat.size(1)
        combined_feat = torch.cat([fused_feat, compare_feat], dim=1)  # [B, S+T, hidden_dim]
        combined_feat = self.transformer_encoder(combined_feat) # [B, S+T, hidden_dim]
            
        # 7. GRU分类
        gru_out, _ = self.gru(combined_feat)  # [B, S+T, hidden_dim*2]
        
        # 全局分类
        global_feat = gru_out[:, -1, :]  # 取最后一个时间步
        logits = self.classifier(global_feat).squeeze(-1)  # [B]
        
        return logits


def count_verification_params(model):
    modules = [
        model.conformer,
        model.feat_proj,
        model.transformer_encoder,
        model.gru,
        model.classifier
    ]
    total = 0
    for m in modules:
        total += sum(p.numel() for p in m.parameters())
    return total

model = MMKWS2(
    text_dim=64,
    audio_dim=1024,
    hidden_dim=128,
    dim=80,
    encoder_dim=128,
    num_encoder_layers=6,
    num_attention_heads=4,
    dropout=0.1,
    num_transformer_layers=2
)
print(f"verification相关参数量: {count_verification_params(model):,}") # 3.5M模型参数量

# if __name__ == "__main__":
#     # 创建一个示例batch
#     batch_size = 2
    
#     # 创建模拟数据
#     anchor_embedding = torch.randn(batch_size, 8, 64)  # 文本嵌入
#     anchor_wave = torch.randn(batch_size, 256, 1024)    # 音频嵌入
#     compare_wave = torch.randn(batch_size, 45, 80)   # Fbank特征
    
#     # 创建长度信息
#     anchor_lengths = torch.LongTensor([8, 6])  # 两个样本的实际长度
#     compare_lengths = torch.LongTensor([45, 40])
    
#     # 创建模型
#     model = MMKWS2(
#         text_dim=64,
#         audio_dim=1024,
#         hidden_dim=128,
#         dim=80,
#         encoder_dim=128,
#         num_encoder_layers=6,
#         num_attention_heads=4,
#         dropout=0.1,
#         num_transformer_layers=2
#     )
    
#     # 计算模型参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"模型总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
    
#     # 打印模型结构
#     print("\n模型结构:")
#     print(model)
    
#     # 模型推理
#     print("\n开始推理...")
#     model.eval()
#     with torch.no_grad():
        
#         print(anchor_embedding.shape)
#         print(anchor_wave.shape)
#         print(compare_wave.shape)
#         print(anchor_lengths.shape)
#         print(compare_lengths.shape)
#         # 完整输入推理
#         logits, seq_logits, text_len = model(
#             anchor_embedding=anchor_embedding,
#             anchor_wave=anchor_wave,
#             compare_wave=compare_wave,
#             anchor_lengths=anchor_lengths,
#             compare_lengths=compare_lengths
#         )
        
#         print("\n推理结果:")
#         print(f"分类logits形状: {logits.shape}")
#         print(f"序列logits形状: {seq_logits.shape}")