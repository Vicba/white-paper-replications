# Word2Vec

Implementation of the Word2Vec model, a popular technique for learning word embeddings. Word2Vec uses neural networks to map words to vectors of real numbers, capturing semantic meanings and relationships between words. This implementation is designed to be educational and straightforward, illustrating the core concepts of the Word2Vec algorithm.

The paper suggests two approaches to implement Word2Vec:

- Continuous Bag of Words (CBoW): predicting the target/center word based on the neighbouring words.
- Skip Gram: predicting the context words based on the center word.

You might be wondering, why are we talking about predicting words here, since our main goal is to compute word embeddings? The trick is, by building a model to predict a word based on its surrounding words, we will find that the weights in the embedding layer are the learned vector representation of the words, which are the embeddings we are looking for.

**The difference between CBOW and Skip-Gram models is in the number of input words. CBOW model takes several words, each goes through the same Embedding layer, and then word embedding vectors are averaged before going into the Linear layer. The Skip-Gram model takes a single word instead.**

resources:
- [The Illustrated Word2vec (Jay alamar)](https://jalammar.github.io/illustrated-word2vec/)
- [Word2vec with PyTorch](https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0)
- [Mastering NLP with PyTorch: Word2Vec](https://medium.com/@patrykmwieczorek/mastering-nlp-with-pytorch-word2vec-60a54030c720)
- [motivation to code this, is from CS224](https://web.stanford.edu/class/cs224n/)

## Features

- **Skip-gram Model**: Trains on the context words to predict the target word.
- **Continuous Bag of Words (CBOW) Model**: Trains on the target word to predict the context words.
- **Customizable Hyperparameters**: Adjustable learning rate, vector size, and window size.


## Parameters

- `vector_size` (int): Dimensionality of the word vectors.
- `window` (int): Maximum distance between the current and predicted word within a sentence.
- `epochs` (int): Number of iterations over the training corpus.


![](https://miro.medium.com/v2/resize:fit:1400/1*7nvAd3P8BDtLx_F7iScHRA.png)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*mLDM3PH12CjhaFoUm5QTow.png)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*eHh1_t8Wms_hqDNBLuAnFg.png)