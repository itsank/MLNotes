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
+ Better then previous representation models, which only use either left context or right context, while Bert uses bidirectional context.
+ Pretrain from unlabeld data using two tasks
  + MLM-> Masked language modeling.
    + 15% of the input tokens are randomly selected and 80% of them are substituted by Mask token, 10% of tem are substituted by random tokens, while rest 10% are unchanged
  + NSP> Next sentence prediction task-> To learn relationship between sentences, predict whether sentence B is actual sentence than. proceeds sentence A, or a random sentences
+ Fine Tune for specific task by adding softmax and linear classifier layers.
  + Used for four type of tasks, Give state-of-the art results.
    - SQuad
    - Glove
    -
    -
    -
+ Uses Self-attention
+ Can handle sequences with different lengths
  + Fixed padding size requiref for pre-training
+ Not suitable for text generation
  + it can do text generation, but are not good in that.



# Interesting papers
#### Embeddings
+ |dsfds|sdfds|
+ |fdsdf|dsfsd|
