# NLP notes

### Transformer
+ Interesting tutotial: Annotated transformer[http://nlp.seas.harvard.edu/2018/04/03/attention.html]
+ Need to have fixed input sequence length, which is differnt for RNN and LSTM based models, which handle variable length sequence.

## Embeddings

#### Word2Vec

#### Glove
+ 

#### ELMO

#### BERT
+ Better then previous representation models, which only use either left context or right context; while Bert uses Msked Language Model
+ MLM
  + 15%% of the input tokens are randomly selected and 80% of them are substituted by Mask token, 10% of tem are substituted by random tokens, while rest 10% are unchanged
+ Pretrain from unlabeld data using two tasks
  + MLM
  + NSP> Next sentence prediction task-> To learn relationship between sentences, predict whether sentence B is actual sentence than. proceeds sentence A, or a random sentences
+ Fine Tuning
  + Used for four type of tasks
    - SQuad
    - Glove
    -
    -
    -
+ Uses Self-attention
+ Can handle sequences with different lengths
  + Fixed padding sze for training

+ Not suitable for text generation



# Interesting papers
#### Embeddings
+ |dsfds|sdfds|
+ |fdsdf|dsfsd|
