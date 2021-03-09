
# Product2Vec implementation (homework assignment 4)

## Pipeline

### Step 1. Prepare data
Make sure that `data/baskets.parquet` exists. Then run:
```shell
source venv/bin/activate
python product2vec/prepare_data.py
```

### Step 2. Train embeddings
```shell
source venv/bin/activate
python product2vec/train_embeddings.py
```

### Step 3. Use embeddings
```python
import torch as t

embedding_in = t.load('data/embedding_in.pt')
embedding_out = t.load('data/embedding_out.pt')
embedding_avg = t.load('data/embedding_avg.pt')
```