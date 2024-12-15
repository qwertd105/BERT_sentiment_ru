from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn

class SentimentBERT(nn.Module):

    class_interpretation = {
        0: "neutral",
        1: "positive",
        2: "negative"
    }

    def __init__(self, load_dir):
        super(SentimentBERT, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.bert(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

        result = {"class": SentimentBERT.class_interpretation[probabilities.argmax().item()],
                   "probabilities": probabilities }
        
        return result