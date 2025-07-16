import os
from dataclasses import dataclass
from pathlib import Path
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from configs import config
from models.spell_check_bert import SpellCheckBert


@dataclass
class TrainingConfig:
    # 训练参数
    epochs: int = 3
    train_batch_size: int = 8
    valid_batch_size: int = 8
    test_batch_size: int = 8
    lr: float = 5e-5
    enable_amp: bool = False

    # 路径相关
    output_dir: Path = Path('./checkpoint')
    logs_dir: Path = Path('./logs')

    # 早停相关
    early_stop_patience: int = 3
    early_stop_metric: str = 'loss'

    # step相关
    log_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 50


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
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=train_config.lr)
        os.makedirs(self.config.output_dir, exist_ok=True)

        # tensorboard
        self.writer = SummaryWriter(log_dir=self.config.logs_dir)
        # 全局step
        self.global_step = 1

        # 早停相关
        self.early_stop_best_score = -float('inf')
        self.early_stop_counter = 0
        # 混合精度训练
        self.scaler = torch.GradScaler(device=self.device.type, enabled=self.config.enable_amp)

    def train(self):
        # 获取数据集
        dataloader = self._get_dataloader(type='train')
        # 训练

        for epoch in range(1, 1 + self.config.epochs):
            for batch_id, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
                # 断点续训
                current_step = (epoch - 1) * len(dataloader) + batch_id
                if current_step < self.global_step:
                    continue

                loss = self._train_step(batch)
                # 保存最优模型
                # 判断是否要保存日志
                if self.global_step % self.config.log_steps == 0:
                    self.writer.add_scalar('loss', loss, self.global_step)
                    tqdm.write(f'[Epoch:{epoch}|step:{self.global_step}] Train Loss:{loss:.4f}')

                # 判断是否要保存checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                # 判断是否要进行评估（早停）
                if self.global_step % self.config.eval_steps == 0:
                    metrics = self.evaluate(type='valid')
                    metrics_str = '|'.join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'[Epoch:{epoch}|step:{self.global_step}] Valid Metrics:{metrics_str}')
                    if self._early_stop(metrics):
                        tqdm.write('early stop')
                        return

                self.global_step += 1

    def evaluate(self, type='test'):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        dataloader = self._get_dataloader(type=type)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=type):
                outputs = self._evaluate_step(batch)
                total_loss += outputs['loss'].item()
                if self.compute_metrics is not None:
                    all_predictions.extend(outputs['predictions'].tolist())
                    all_labels.extend(batch['labels'].tolist())
        # 统计评估结果
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(all_predictions, all_labels)
        else:
            metrics = {}
        metrics['loss'] = total_loss / len(dataloader)
        return metrics

    def _evaluate_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def _train_step(self, batch):
        self.model.train()
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,enabled=self.config.enable_amp):
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

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

    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'early_stop_best_score': self.early_stop_best_score,
            'early_stop_counter': self.early_stop_counter
        }
        torch.save(checkpoint, self.config.output_dir / 'checkpoint.pt')

    def _load_checkpoint(self):
        checkpoint_path = self.config.output_dir / 'checkpoint.pt'
        if checkpoint_path.exists():
            print("检查点存在，开始加载")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.early_stop_best_score = checkpoint['early_stop_best_score']
            self.early_stop_counter = checkpoint['early_stop_counter']
        else:
            print("检查不存在，从头训练")

    def _early_stop(self, metrics):
        score = metrics[self.config.early_stop_metric]
        if self.config.early_stop_metric == 'loss':
            score = -score

        if score > self.early_stop_best_score:
            self.early_stop_best_score = score
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), self.config.output_dir / 'best.pt')
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.config.early_stop_patience:
                return True
            else:
                return False


if __name__ == '__main__':
    model = SpellCheckBert()
    dataset_dict = load_from_disk(str(config.DATA_DIR / 'spell_check/processed/bert'))
    training_config = TrainingConfig(output_dir=config.CHECKPOINT_DIR / 'spell_check_bert',
                                     logs_dir=Path('/Users/zhangyf/PycharmProjects/nlp/graph/logs'),
                                     log_steps=50,
                                     save_steps=1000,
                                     eval_steps=500,
                                     epochs=30)
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR)


    def compute_metrics(predictions, labels):
        # predictions : [[1,3,5],[2,3,6],[1,2,3],[7,0,9]]
        # labels :[[1,3,5],[2,4,6],[1,2,3],[7,8,9]]
        total_count = 0
        correct_count = 0
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        for pred, label in zip(predictions, labels):
            if pred == label:
                correct_count += 1
            total_count += 1
        return {'accuracy': correct_count / total_count}


    trainer = Trainer(model,
                      # dataset_dict['train'].select(range(100)),
                      dataset_dict['train'],
                      dataset_dict['valid'],
                      dataset_dict['test'],
                      training_config,
                      compute_metrics=compute_metrics)
    trainer.train()
