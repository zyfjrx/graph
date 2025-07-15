import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from models.spell_check_bert import SpellCheckBert

@dataclass
class TrainingConfig:
    epochs: int = 3
    train_batch_size: int = 8
    valid_batch_size: int = 8
    test_batch_size: int = 8
    lr: float = 5e-5
    output_dir: Path = Path('./checkpoint')
    logs_dir: Path = Path('./logs')
    enable_amp: bool = True
    early_stop_patience: int = 3
    early_stop_metric: str = 'loss'


class Trainer:
    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 train_config,
                 compute_metrics=None,
                 optimizer=None
                 ):
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = train_config
        self.compute_metrics = compute_metrics
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=train_config.lr)
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.best_loss = float('inf')

    def train(self):
        # 获取数据集
        dataloader = self._get_dataloader(type='train')
        # 训练
        self.model.train()
        for epoch in range(1, 1 + self.config.epochs):
            print(f"========== Epoch {epoch}/{self.config.epochs} ==========")
            avg_loss = self._train_one_epoch(dataloader)
            print(f"avg_loss: {avg_loss}")

            if avg_loss < self.best_loss:
                print("误差减小了，保存模型 ...")
                self.best_loss = avg_loss
                torch.save(self.model.state_dict(), self.config.output_dir / 'best.pt')
                print("保存成功")
            else:
                print("无需保存模型 ...")

    def evaluate(self):
        pass

    def _train_one_epoch(self, dataloader):
        total_loss = 0
        for batch in tqdm(dataloader, desc='train'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss / len(dataloader)

    def _get_dataloader(self, type):
        if type == 'train':
            dataset = self.train_dataset
            batch_size = self.config.train_batch_size
        elif type == 'valid':
            dataset = self.valid_dataset
            batch_size = self.config.valid_batch_size
        elif type == 'test':
            dataset = self.test_dataset
            batch_size = self.config.test_batch_size
        else:
            raise ValueError('Invalid dtype')

        dataset.set_format(type='torch')
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    model = SpellCheckBert()
    dataset_dict = load_from_disk(str(config.DATA_DIR / 'spell_check/processed/bert'))
    training_config = TrainingConfig(output_dir=config.CHECKPOINT_DIR / 'spell_check_bert')
    trainer = Trainer(model,
                      # dataset_dict['train'].select(range(100)),
                      dataset_dict['train'],
                      dataset_dict['valid'],
                      dataset_dict['test'],
                      training_config)
    trainer.train()
