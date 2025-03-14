# Transformer Classification Implementation from Scratch
This excercise involves the implmeentation from scratch of the following architecture:
- Token Embeddings: Convert input tokens into vector representations.
- Encoder block
    - Positional Encoding: Since transformers don’t have recurrence, we need to add positional information.
    - Multi-Head Self-Attention: Core mechanism for attending to different parts of the sequence.
    - Feedforward Layer: A simple MLP after attention to refine the representations.
    - Layer Normalization & Residual Connections: For stable training.
- Classification head

## Dataset
(https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity.

### Train set
The file `ag_news_train.csv` consists of 120,000 training samples of news articles that contain 3 columns. The first column is Class Id, the second column is Title and the third column is Description. The class ids are numbered 1-4 where 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.


### Train set
The file `ag_news_test.csv` consists of 7600 testing samples of news articles that contain 3 columns. The first column is Class Id, the second column is Title and the third column is Description. The class ids are numbered 1-4 where 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.


## Params
### Model Configuration
```
vocab_size = 30522  # BERT's vocab size
embed_dim = 512
num_heads = 8
ff_hidden_dim = 1024
num_layers = 3
max_seq_len = 50
num_classes = 4  # AG News has 4 classes
```

###  Training Configuration
```
batch_size = 32
max_epochs = 100  # Upper limit, but early stopping may stop earlier
early_stopping_threshold = 0.02  # Stop if loss improvement < threshold
patience = 3  # Number of epochs to wait before stopping if no improvement
learning_rate = 3e-4
```

### Hyperparam tuning
The hyperparameters were searched among predefined sets of values designed to achieve small resulting models to mitigate overfitting after a first experimental run showing good train-set-loss, but validation loss degradation.
```
configs = [
    {"n_heads": 4, "num_layers": 2, "ff_hidden_dim": 512, "embed_dim": 256},  # Smallest model
    {"n_heads": 4, "num_layers": 2, "ff_hidden_dim": 1024, "embed_dim": 256},  # Larger FFN
    {"n_heads": 4, "num_layers": 3, "ff_hidden_dim": 512, "embed_dim": 256},  # More layers
    {"n_heads": 8, "num_layers": 2, "ff_hidden_dim": 512, "embed_dim": 512},  # Larger embed
    {"n_heads": 8, "num_layers": 3, "ff_hidden_dim": 512, "embed_dim": 512},  # More layers
    {"n_heads": 8, "num_layers": 3, "ff_hidden_dim": 1024, "embed_dim": 512},  # Largest
]
```

## Results
The model appears to be too powerful for the selected dataset, as the training loss keeps diminishing while the validation loss reaches its minimum at epoch 4 before increasing, showing sings of overfitting.

Sample training curve:

<img src="https://github.com/user-attachments/files/19247372/loss_plot_6_heads8_layers3_ff1024_embed512.pdf" width="48">


## Conclusion
The model appears to be too powerful for the dataset used, resulting in substantial overfitting.
Further optimization of the model is beyond the scope of this excercise, but optimization measures are listed below.

### Next steps
Fistly, compare the trained model to a baseline with a simpler architecture. If the models perform similarly, our Transformer may be too complex for the task.

With the goal of constraining the model's learning to better generalize to unseen cases, we can do the following:
1. Increase regularization (increase dropout, )
Train the model on larger datasets
2. Use pre-trained transformer encoder weights as a starting point, freeze layers and finetune final layer(s) on dataset at hand
3. Keep decresing model size (decrease `n_heads`, `num_layers`, `ff_hidden_dim`, `embed_dim` hyperparams)
4. Apply local attention: instead of attending to all tokens, restrict attention to a local window (e.g., 5-10 tokens)
