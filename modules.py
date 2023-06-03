import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

class CollateFn:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def collate(self, batch):
        inputs = [instance['document'] for instance in batch]
        targets = [instance['summary'] for instance in batch] if 'summary' in batch[0] else None

        enc_inputs = self.tokenizer.encode_batch(inputs)
        if targets is not None: enc_targets = self.tokenizer.encode_batch(targets)

        input_ids = pad_sequence([torch.tensor(input.ids) for input in enc_inputs], batch_first=True, padding_value=self.tokenizer.token_to_id("[PAD]"))
        attention_mask = pad_sequence([torch.tensor(input.attention_mask) for input in enc_inputs], batch_first=True, padding_value=0).to(torch.bool)
        assert input_ids.shape == attention_mask.shape
        if targets is not None: target_ids = pad_sequence([torch.tensor(target.ids) for target in enc_targets], batch_first=True, padding_value=self.tokenizer.token_to_id("[PAD]"))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "targets": target_ids}