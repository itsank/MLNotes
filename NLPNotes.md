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
    
#### Joulin el al, Bag of Tricks [Aug 2016 pdf](https://arxiv.org/pdf/1607.01759.pdf)
+ Introduced FastText classifier, results are often on par with deep learning based classifeirs, much much faster can train 1Billon words in a single cpu in 10 mins. Very good results of that time. Showed using n-gram up to 5 leads to best performance.
+ Explored ways to sclae linear classifiers to large corpus, using simple two layer NN. 
+ Analysied their work on tag prediction and sentiment analysis.
+ Used Hierarchical softmax based on Huffman code tree to speed up classification. Complexity drop from O(kn) to O(hlogk)m (k no of classes, h dimention of text representation). They performed depth first search and tracked maximum probability among leafs to discard small probability branches, thus speeding up.
+ Used Bag of n-grams as features 
+ Used hashin tricks for efficient mapping of n-grams.
+ It outperform LSTM and deep model becuase fasttext get entire paragaph at pnce, while LSTm and other methord are sequencial,may be deep model overfit more frequently as the model complexity is very high. May LSTM forget the earlier part of the text, that may be the reason why it didn't perform better.
+ They evaluated their model on the test domain which are not good for comparing with large models.
+ It capture word order information to some extend using bigram and ngrams.

#### Tang, el al, PTE: Predictive Text Embedding Through Large-Scale Heterogeneous Text Networks, Aug'15 [pdf](https://arxiv.org/pdf/1508.00200.pdf)
+ PTE utilizes both labeled and unlabeled to learn text embeddings.
+ labeled word co-occurrence,first represented as a large-scale heterogeneous text network, then embedded into a low dimensional space through an efficient algorithm.
+ Redcues Data sparseness through test Embedding
+ Use three level to network to map local context, document and category level
+ Model Bipartite Network Enbedding, preserve the first-order and second order proximity.



    
# Interesting papers
#### Embeddings

Attention
+ High generalization Capacity
+ Surprising performance yet simple structure
+ Still didn't fully understand why attention world
Transformer (Attention is all you need)
+ 

#### Semi-Supervised Learning and Active Learning

Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding [pdf](https://papers.nips.cc/paper/5849-semi-supervised-convolutional-neural-networks-for-text-categorization-via-region-embedding.pdf)
+ Uses convolutional neural networks (CNNs) for text categorization
+ Unlike the previous approaches that rely on word embeddings, our method learns embeddings of small text regions from unlabeled data for integration into a supervised CNN.
+ Create embedding for n-gram
+ TV embedding (Try to predict context both forward and backward), these left and right context are two views, they are working on n-gram instead of a single word.
+ cnn produces non-linear embedding
+ Train a CNN to perfom in two task 1) skipgram and 2) classification
+ Used relu activateion
+ Most Semi-supervised appratches in NLP uses Word2vec embedding, which is made with out any contextual 
+ Co-learning, create multiple views of the data, take that view which give good result.
+ CNN can get high level concepts from text
+ Using regon of the word allwos rge cnn to extract concepts bettwer


Active Deep Networks for Semi-Supervised Sentiment Classification [pdf](https://www.aclweb.org/anthology/C10-2173)
+ Used for sentiment classification
+ Active Learning:- Actively select the label data and unlabel data
+ Use RBMS, energy based models, hgher the energy the lower is the compatiility.
+ Preprocesing, each review is presented as one-hot vector ; punchtation and number and word of length 1 are removed.; Top 1.5 % words are removed.
+ Training RBM steps 1) Initailize weight 2) samplying the weight m times using Gibbs sampling 3) Contrastive divergence
+ Active learning (ADN)
    + select ambigous data points, select data points near the seperating hyper planes
    + Select equal number of + and -ve instances we get the better performance.
+ Require limited number of training data
+ Choose data to close to hyperplane to be labeled while training
+ Two stage -> 1) stage using RBM 2) supervised learning using active learning framework; both steps are done iteratively.







