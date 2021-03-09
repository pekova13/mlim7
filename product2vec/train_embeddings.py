
import sys
sys.path.append('.')

import torch as t

from product2vec.classes import BasketsStreamer, Product2VecTrainer
from product2vec.config import config

PREPARED_BASKETS = 'data/baskets_for_p2v.pkl'


streamer = BasketsStreamer(**config['streamer'])
streamer.load_baskets(PREPARED_BASKETS)

trainer = Product2VecTrainer(streamer=streamer, **config['trainer'])
trainer.train()
trainer.plot_loss()

embedding_in = trainer.get_embedding('in')
embedding_out = trainer.get_embedding('out')
embedding_avg = trainer.get_embedding('avg')

t.save(embedding_in, 'data/embedding_in.pt')
t.save(embedding_out, 'data/embedding_out.pt')
t.save(embedding_avg, 'data/embedding_avg.pt')
