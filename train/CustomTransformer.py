import torch
import torch.nn as nn
from transformers import AutoModel

class AutoModelForSequenceClassification(nn.Module):
    def __init__(self, checkpoint, num_labels ):
        super(AutoModelForSequenceClassification, self).__init__()
        self.num_labels = num_labels

        self.model = AutoModel.from_pretrained(checkpoint, )
                                              #  config = AutoConfig.from_pretrained(
                                              #      checkpoint,
                                              #      output_attention = True,
                                              #      output_hidden_state = True ) )

        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids = None, attention_mask=None, labels = None ):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask  )
        last_hidden_state = outputs[0]
        x = self.classifier(last_hidden_state[:, 0, : ].view(-1, 768 ))
        return x
