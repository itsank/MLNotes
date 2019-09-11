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
+ Uses Self-attention
+ Can handle sequences with different lengths
  + Fixed padding size requiref for pre-training
+ Not suitable for text generation
  + it can do text generation, but are not good in that.


## Text Classification

### Generative vs Discriminative Models
+ Model the distribution of the outline data, to do generation of data similar to inout data. Model joint distribution
### Discriminative Models
+ Learn boundaries between classes in some latent space, thus these model learn condition probability distribution.

Tips


+ Termination citeria:- 
    + Number of epochs
    + Threshold on training set errors, 
    + No descrese in error or increase validation error.
    
#### Joulin el al, Bag of Tricks [Aug 2016 pdf] (https://arxiv.org/pdf/1607.01759.pdf)
+ Introduced FastText classifier, results are often on par with deep learning based classifeirs, much much faster can train 1Billon words in a single cpu in 10 mins. Very good results of that time. Showed using n-gram up to 5 leads to best performance.
+ Explored ways to sclae linear classifiers to large corpus, using simple two layer NN. 
+ Analysied their work on tag prediction and sentiment analysis.
+ Used Hierarchical softmax based on Huffman code tree to speed up classification. Complexity drop from O(kn) to O(hlogk)m (k no of classes, h dimention of text representation). They performed depth first search and tracked maximum probability among leafs to discard small probability branches, thus speeding up.
+ Used Bag of n-grams as features 
+ Used hashin tricks for efficient mapping of n-grams.


    
# Interesting papers
#### Embeddings

