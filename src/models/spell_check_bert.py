from torch import nn
from transformers import AutoTokenizer, BertModel

class SpellCheckBert(nn.Module):
    def __init__(self):
        super(SpellCheckBert, self).__init__()
        self.bert = BertModel.from_pretrained("/Users/zhangyf/llm/bert-base-chinese")
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.bert.config.pad_token_id)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # logits.shappe = [batch_size, seq_len, vocab_size]
        logits = self.linear(outputs.last_hidden_state)
        loss = 0.0
        if labels is not None:
            loss += self.loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        return {"loss": loss, "logits": logits}

