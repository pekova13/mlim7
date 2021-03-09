
import pandas as pd
import sys

sys.path.append('.')

from product2vec.classes import BasketsStreamer
from product2vec.config import config

DF_PATH = 'data/baskets.parquet'
PREPARED_BASKETS = 'data/baskets_for_p2v.pkl'


baskets = pd.read_parquet(DF_PATH)

streamer = BasketsStreamer(**config['streamer'])
streamer.load_data(baskets)
streamer.save_baskets(PREPARED_BASKETS)
