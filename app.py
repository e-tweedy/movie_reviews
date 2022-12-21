import re
import torch
import torchdata
from torch import nn
import gradio as gr
import torchtext

# Import the saved vocabulary
vocab=torch.load('imdb_rnn_drop_model_lr_1e-2_vocab.pth',map_location=torch.device('cpu'))

"""A function which removes html tags, standardized emoticons ( :-) --> :) , etc.), and tokenizes by word."""
def tokenizer(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower()
    )
    text = (re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-',''))
    tokenized = text.split()
    return tokenized

"""A Recurrent Neural Network which has an embedding layer, LSTM layer, and then three FC layers with dropout layers in-between."""
class RNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,rnn_hidden_size,dc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.rnn = nn.LSTM(embed_dim,rnn_hidden_size,batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size,fc_hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(rnn_hidden_size,fc_hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(fc_hidden_size,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,text,lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out,lengths.cpu().numpy(),enforce_sorted=False,batch_first=True)
        out,(hidden,cell) = self.rnn(out)
        out = hidden[-1,:,:]
        out=self.fc1(out)
        out=self.relu1(out)
        out=self.dropout1(out)
        out=self.fc2(out)
        out=self.relu2(out)
        out=self.dropout2(out)
        out=self.fc3(out)
        out = self.sigmoid(out)
        return out

# Initialize our model
vocab_size=len(vocab)
embed_dim=20
rnn_hidden_size=64
fc_hidden_size=64

torch.manual_seed(1)
model = RNN(vocab_size,embed_dim,rnn_hidden_size,fc_hidden_size)

# Imprint this model with the state_dict from the trained model
model.load_state_dict(torch.load('imdb_rnn_drop_model_lr_1e-2_weights.pth',map_location=torch.device('cpu')))
model.eval()

"""Prediction function.  Tokenizes input text string, maps with the vocab to token indices, and then sends to the model to predict 'Positive' or 'Negative' sentiment."""
text_pipeline=lambda x: [vocab[token] for token in tokenizer(x)]

def predict(text):
    text_list, lengths = [],[]
    processed_text=torch.tensor(text_pipeline(text),dtype=torch.int64)
    text_list.append(processed_text)
    lengths.append(processed_text.size(0))
    lengths=torch.tensor(lengths)
    padded_text_list=nn.utils.rnn.pad_sequence(text_list,batch_first=True)
    padded_text_list = padded_text_list
    length = lengths
    pred = model(padded_text_list,lengths)
    
    return 'Positive' if pred > 0.5 else 'Negative'

# Initialize the gradio interface
title = "Write a movie review"
description = "Enter a review for a movie you've seen.  This tool will try to guess whether your review is positive or negative."
gr.Interface(fn=predict, 
             inputs="text",
             outputs="label",
             title = title,
             description = description,
              ).launch()

