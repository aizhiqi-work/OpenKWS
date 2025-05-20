import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from model_pinyin import MMKWS2
from torch.utils.data import Dataset, ConcatDataset

class MMKWS2_Wrapper(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MMKWS2()
        self.criterion = nn.BCEWithLogitsLoss()
        # 将hubert_model设为临时变量而非类属性
        hubert_model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-large").half().eval()
        self._hubert_model = hubert_model  # 使用下划线前缀表示内部使用
        
    def training_step(self, batch, batch_idx):
        anchor_wave, anchor_text_embedding, compare_wave, compare_lengths, label, seq_label = \
            batch['anchor_wave'], batch['anchor_embedding'], batch['compare_wave'], batch['compare_lengths'], batch['label'], batch['seq_label']
            
        with torch.no_grad():
            outputs = self._hubert_model(anchor_wave.half())
            anchor_wave_embedding = outputs.last_hidden_state
        
        anchor_wave_embedding = anchor_wave_embedding.to(anchor_wave.dtype)
        
        logits, seq_logits = self.model(
            anchor_wave_embedding,
            anchor_text_embedding,
            compare_wave,
            compare_lengths
        )
        
        # 句级二分类loss
        utt_loss = self.criterion(logits, label.float())
        
        # 序列loss（mask掉seq_label为-1的部分）
        mask = (seq_label != -1).float()
        seq_label_valid = seq_label.clone()
        seq_label_valid[seq_label == -1] = 0  # 避免-1影响loss
        
        seq_loss = F.binary_cross_entropy_with_logits(
            seq_logits, seq_label_valid.float(), weight=mask, reduction='sum'
        ) / (mask.sum() + 1e-6)
        
        loss = utt_loss + seq_loss
        
        # 每500步记录日志
        self.log('train/utt_loss', utt_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/seq_loss', seq_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        def lr_lambda(step):
            return 0.95 ** (step // 1000)
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lr_lambda
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 按步更新
                "frequency": 1
            },
        }


# 3. 设置 Trainer 和训练
if __name__ == "__main__":
    pl.seed_everything(2024)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    from dataset_pinyin import PairDataset
    from dataloader_pinyin import get_dataloader
    
    # 创建数据集
    dataset1 = PairDataset(
        '/nvme01/aizq/kws-agent/data/anchor_pairs.parquet', 
        '/nvme01/aizq/kws-agent/data/WenetPhrase_base/M_S',
        augment=True
    )
    dataset2 = PairDataset(
        '/nvme01/aizq/kws-agent/data/synthetic_pairs.parquet', 
        '/nvme01/aizq/kws-agent/data/WenetPhrase_base/M_S',
        augment=True
    )
    dataset = ConcatDataset([dataset1, dataset2])
    # 创建dataloader
    dataloader = get_dataloader(dataset, batch_size=1024)

    model = MMKWS2_Wrapper()
    model_checkpoint = ModelCheckpoint(
        dirpath="/nvme01/aizq/kws-agent/ckpts",
        filename="step{step:06d}",
        save_top_k=-1,
        save_on_train_epoch_end=False,  # 按训练步数保存
        every_n_train_steps=500       # 每10k步保存一次
    )
    logger = pl.loggers.TensorBoardLogger('/nvme01/aizq/kws-agent/logs/', name='MMKWS+')
    trainer = Trainer(
        devices=1, 
        accelerator='gpu', 
        logger=logger,
        max_epochs=4,  # 训练1轮
        callbacks=[model_checkpoint], 
        accumulate_grad_batches=2, # 2048 batchsize
    )
    trainer.fit(model, train_dataloaders=dataloader)