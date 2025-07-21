import torch
from transformers import AutoTokenizer
from models.spell_check_bert import SpellCheckBert
from models.spell_check_t5 import SpellCheckT5

class BasePredictor:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def predict(self, inputs: list[str] | str):
        pass



class SpellCheckBertPredictor(BasePredictor):

    def predict(self, inputs: list[str] | str):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        # 处理输入数据
        inputs = self.tokenizer(inputs, padding='max_length', truncation=True, return_tensors="pt", max_length=64)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        predictions = outputs['predictions']
        batch_result = self.tokenizer.batch_decode(predictions, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        batch_result = [result.replace(' ', '') for result in batch_result]
        if is_str:
            return batch_result[0]
        return batch_result



class SpellCheckT5Predictor(BasePredictor):

    def predict(self, inputs: list[str] | str):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        # 处理输入数据
        inputs = self.tokenizer(inputs, padding='max_length', truncation=True, return_tensors="pt", max_length=64)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        predictions = self.model.generate(input_ids, attention_mask=attention_mask)
        batch_result = self.tokenizer.batch_decode(predictions, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if is_str:
            return batch_result[0]
        return batch_result

if __name__ == '__main__':
    from configs import config
    model = SpellCheckT5()
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / 'spell_check_t5' / 'best.pt',map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained("/Users/zhangyf/llm/mengzi-t5-base")
    predict = SpellCheckT5Predictor(model, tokenizer,device='cpu')
    print(predict.predict(['许多迢害着食军人。']))