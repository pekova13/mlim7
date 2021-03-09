
import torch as t
import sys
sys.path.append('.')

from product2vec.classes import ProductMapper
from product2vec.config import config


embedding = t.load('data/embedding_avg.pt')

mapper = ProductMapper(**config['mapper'])
mapper.fit(embedding)
mapper.plot()