# GloVe

My implementation/understanding of GloVe (Global Vectors for Word Representation), a popular technique used in natural language processing (NLP) for creating word embeddings.

While the concept of the co-occurrence matrix was straightforward, it took me a very long time to fully grasp the working/details of the GloVe paper and its implementation.

Also learned about xavier initialization in this one :).

some helpful resources:
- [NLP — Word Embedding & GloVe](https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6)
- [GloVe](https://medium.com/@ellie.arbab/glove-8849a40c08bc)
- [Intuitive Guide to Understanding GloVe Embeddings](https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010)
- [pytorch-glove github (noaRricky)](https://github.com/noaRricky/pytorch-glove)
- [GloVe github (daandouwe)](https://github.com/daandouwe/glove/tree/master)
- going through some yt videos
- bit of chatgpt

# Usage

1. make venv and install requirements
```bash
python -m venv venv

# mac
source venv/bin/activate
# windows
venv\Scripts\activate

pip install -r requirements.txt
```

2. run train.py
```bash
python train.py
```