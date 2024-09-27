# BERT

BERT is a model developed by Google that uses a transformer architecture to understand the context of words bidirectionally which improves multiple natural language processing tasks.

I coded the model first by myself wich was very easy but then struggled for days on letting in pretrain with next sentence prediction and masked language modeling. Thats why a lot of surrounding code (dataset, trainer, preprocess) comes from the last bullet point in sources.

Sources for this code:
- [YT BERT explaiend, Umar Jamil](https://www.youtube.com/watch?v=90mGPxR2GgY)
- [YT BERT, CodeEmporium](https://www.youtube.com/watch?v=xI0HHN5XKDo)
- [Mastering BERT Model: Building it from Scratch with Pytorch](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)
- [How to Code BERT Using PyTorch – Tutorial With Examples](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial)
- [BERT-pytorch (github dreamgonfly)](https://github.com/dreamgonfly/BERT-pytorch/tree/master)
- [BERT-pytorch (github codertimo)](https://github.com/codertimo/BERT-pytorch/tree/master)
- [DataScience NLP (github ChanCheeKean)](https://github.com/ChanCheeKean/DataScience/blob/main/13%20-%20NLP/C04%20-%20BERT%20(Pytorch%20Scratch).ipynb)

## Usage

1. make venv and install requirements

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

2. get latest data by running data_setup.sh

```bash
./scripts/data_setup.py
```

3. pretrain

```bash
python trainer.py
```
