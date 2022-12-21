# movie_reviews
A Recursive Neural Network (RNN) model for predicting the sentiment (positive or negative) of movie reviews.  The RNN was trained on a dataset of IMDB movie reviews.  Includes Gradio web app implementation.  Code is implemented in Python using PyTorch libraries.

The prediction is made using a RNN which was trained on the IMDB review dataset built into torchtext.datasets : https://pytorch.org/text/stable/datasets.html#imdb The network architecture is as follows:

RNN(

(embedding): Embedding(75979, 20, padding_idx=0)

(rnn): LSTM(20, 64, batch_first=True)

(fc1): Linear(in_features=64, out_features=64, bias=True)

(relu1): ReLU()

(dropout1): Dropout(p=0.5, inplace=False)

(fc2): Linear(in_features=64, out_features=64, bias=True)

(relu2): ReLU()

(dropout2): Dropout(p=0.5, inplace=False)

(fc3): Linear(in_features=64, out_features=1, bias=True)

(sigmoid): Sigmoid()

)

The 75979-word vocabulary was constructed by word-tokenizing the IMDB training set.

The files in this repository are as follows:

imdb_sentiment_dropout.ipynb - a Jupyter notebook demonstrating the process of importing the datasets, creating and training the model, and feeding predictions into a Gradio web app.

imdb_rnn_drop_model_lr_1e-2_vocab.pth - the vocab of the trained model

imdb_rnn_drop_model_lr_1e-2_weights.pth - the weights of the trained model

app.py - the Python file which can be used to implement the Gradio app on Hugging Face or similar.


This app can be found implemented and hosted at this link: https://huggingface.co/spaces/etweedy/movie_review
