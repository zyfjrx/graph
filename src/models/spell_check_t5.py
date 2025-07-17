import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class SpellCheckT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.t5 = AutoModel.from_pretrained("/Users/zhangyf/llm/mengzi-t5-base")
        self.linear = nn.Linear(self.t5.config.hidden_size, self.t5.config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.t5.config.pad_token_id)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        :param input_ids: 原始序列
        :param attention_mask: 原始序列mask
        :param labels: 目标序列
        :return:
        """

        # 处理解码器的输入
        decoder_input_ids = self.t5._shift_right(labels)
        outputs = self.t5(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids)
        # logits.shappe = [batch_size, seq_len, vocab_size]
        logits = self.linear(outputs.last_hidden_state)
        predictions = torch.argmax(logits, dim=-1)
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        return {"loss": loss, "predictions": predictions}

