from collections import defaultdict
import datetime
import itertools
import math
import pickle
import random
import timeit
from tqdm import tqdm

from sklearn.manifold import TSNE

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT

from plotly.offline import iplot
import plotly.graph_objs as go


class BasketsStreamer:
    """ An iterable data streamer
    
    Usage:
    >> streamer = BasketsStreamer(**kwargs)
    >> streamer.load_data(data=data) # load data from pandas dataframe
    >> streamer.save_baskets()       # save prepared baskets
    >> streamer.load_baskets()       # load prepared baskets
    
    >> streamer.rewind()             # reset iteration 
    >> for center, context in streamer:
    ..     pass
    
    Kwargs:
        batch_size: the size of a single batch
        shuffle:    whether baskets should be returned in random order
        sample:     fraction of baskets (e.g. 0.5) to be randomly drawn
        verbose:    whether to print progress
        basket:     name of the column containing basket IDs
        product:    name of the column containing product IDs
    
    Returns at each iteration:
        center:     tuple of size `batch_size` containing product IDs
        context:    tuple of size `batch_size` containing product IDs
    """
    
    def __init__(self,
                 batch_size: int = 10_000, 
                 shuffle: bool = True,
                 sample: float = 1.0,
                 verbose: bool = True,
                 basket: str = 'basket',
                 product: str = 'product',
                ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample = sample
        self.verbose = verbose
        self.BASKET = basket
        self.PRODUCT = product
        
        self._data = None
        self._baskets = None
        
        assert self.sample <= 1.0
        
    def load_data(self, data: pd.DataFrame = None) -> None:
        """
        Loads data from a pandas data frame and prepares it for Product2Vec.
        """
        self._data = data
        self._fill_baskets()
        self._fill_cache()
        
    def _fill_baskets(self) -> None:
        """
        Creates a dictionary `._baskets` containing lists of products contained in each basket.
        """
        self._baskets = defaultdict(lambda: list())
        self._nr_products = len(set(self._data[self.PRODUCT]))
        
        if self.verbose:
            row_iterator = tqdm(self._data.iterrows(), total=self._data.shape[0])
            row_iterator.set_description('Filling the baskets')
        else:
            row_iterator = self._data.iterrows()
        
        for _, row in row_iterator:
            self._baskets[row[self.BASKET]].append(row[self.PRODUCT])
            
        self._baskets = dict(self._baskets)
        
    def _fill_cache(self):
        """
        Creates a list `._cache` containing pairs of (center, context) product IDs
        for each existing permutation based on `._baskets`.
        """
        self._cache = []
        
        if self.shuffle:
            baskets = list(self._baskets) # type: ignore
            random.shuffle(baskets) # inplace
        else:
            baskets = self._baskets
            
        if self.sample < 1.0:
            baskets = list(self._baskets) # type: ignore
            k = math.ceil(len(baskets) * self.sample)
            baskets = random.sample(baskets, k=k)
        
        if self.verbose:
            basket_iterator = tqdm(baskets)
            basket_iterator.set_description('Filling the cache  ')
        else:
            basket_iterator = self._baskets
            
        for basket in basket_iterator:
            self._cache += list(itertools.permutations(self._baskets[basket], 2))
            
        self._batches_returned = 0
        self._batches_total = math.ceil(len(self._cache) / self.batch_size)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._batches_returned < self._batches_total:
            start = self._batches_returned * self.batch_size
            out = self._cache[start : start + self.batch_size]
            center, context = list(zip(*out))
            self._batches_returned += 1
            return center, context
        else:
            raise StopIteration
            
    def rewind(self):
        """
        Rewinds (resets) the iterator.
        """
        self._batches_returned = 0
        
    def save_baskets(self, path='baskets_streamer.pkl'):
        """
        Saves the prepared baskets from `._baskets` to path.
        """
        with open(path, "wb") as f:
            pickle.dump((self._baskets, self._nr_products), f) 
    
    def load_baskets(self, path='baskets_streamer.pkl'):
        """
        Loads the previously prepared baskets from path to `._baskets`.
        Can be used instead of `.load_data()` to spead up the process.
        """
        with open(path, "rb") as f:
            self._baskets, self._nr_products = pickle.load(f)
        self._fill_cache()


class Product2Vec(nn.Module):
    """ A torch implementation of Skipgram algorithm with negative sampling
    
    Adjusted from https://github.com/theeluwin/pytorch-sgns
    """

    def __init__(self, vocab_size: int, embedding_size: int, padding_idx: int = 0):
        super(Product2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        tmp = 0.5 / self.embedding_size
        self.ivectors.weight = nn.Parameter(FT(self.vocab_size, self.embedding_size).uniform_(-tmp, tmp))
        self.ovectors.weight = nn.Parameter(FT(self.vocab_size, self.embedding_size).uniform_(-tmp, tmp))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = (t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True)
                      .view(batch_size, -1))
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        #oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1) # throws an error
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean()
        nloss = (t.bmm(nvectors, ivectors).squeeze().sigmoid().log()
                 .view(-1, context_size, self.n_negs).sum(2).mean(1))
        return -(oloss + nloss).mean()
    

class Timer:
    """Minimal timer
    
    Usage:
    >> timer = Timer()
    >> print(f'Time elapsed: {timer}')
    
    Reference: https://stackoverflow.com/a/57931660/
    """
    def __init__(self, round_ndigits: int = 0):
        self._round_ndigits = round_ndigits
        self._start_time = timeit.default_timer()

    def __call__(self) -> float:
        return timeit.default_timer() - self._start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), self._round_ndigits)))


class Product2VecTrainer:
    """ Wrapper class to train Product2Vec with data from BasketsStreamer
    
    Kwargs:
        streamer:       instance of BasketsStreamer
        embedding_size: number of embedding dimensions
        epochs:         number of epochs
        early_stop:     a minimal train loss improvement 
                        (if less than this value, algorithm stops early)
        learning_rate:  optimization learning rate
        n_negative:     number of negative samples
        test_share:     share of batches to be used for testing
                        (approx. equal to the share of baskets used for testing)
        verbose:        whether to show stats for each epoch
        
    Usage:
    >> trainer = Product2VecTrainer(streamer=streamer, **kwargs)
    >> trainer.train()
    """
    
    def __init__(
            self,
            streamer: BasketsStreamer,
            embedding_size: int = 300,
            epochs: int = 10,
            early_stop: float = 0.01,
            learning_rate: float = 0.001,
            n_negative: int = 2,
            test_share: float = 0.2,
            verbose: bool = True
            ) -> None:
        self.streamer = streamer
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.n_negative = n_negative
        self.test_share = test_share
        self.verbose = verbose
        self.nr_products = self.streamer._nr_products
        self.nr_batches = self.streamer._batches_total
        self.nr_batches_train = math.ceil(self.nr_batches * (1 - self.test_share))
        self.nr_batches_test = self.nr_batches - self.nr_batches_train
        self.batch_size = self.streamer.batch_size

        self.p2v = Product2Vec(vocab_size=self.nr_products, embedding_size=self.embedding_size)
        self.sgns = SGNS(embedding=self.p2v, vocab_size=self.nr_products, n_negs=self.n_negative)
        self.optim = t.optim.Adam(self.sgns.parameters(), lr=self.learning_rate)
        
        # parallelization
        # reference: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        #device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        #available_gpu = t.cuda.device_count()
        #if available_gpu > 1:
        #    print(f'Using {available_gpu} GPUs.')
        #    self.p2v = nn.DataParallel(self.p2v)
        #else:
        #    print(f'Using {device.type}.')
        #self.p2v.to(device)
        
    def train(self):
        """
        Trains the embedding.
        """
        self.loss_train = []
        self.loss_test = []
        epoch_iterator = range(self.epochs)
        
        if self.verbose:
            tqdm._instances.clear() # reset any existing progress bars
            pbar = tqdm(total=self.nr_batches*self.epochs)
            
        for epoch in range(self.epochs):
            if self.verbose:
                pbar.set_description(f'Epoch {(epoch+1):2}')
            l_train = 0
            l_test = 0
            
            self.streamer.rewind()
            for idx, (center, context) in enumerate(self.streamer):
                center, context = t.tensor(center), t.tensor(context).reshape(-1, 1)
                loss = self.sgns(center, context)
                
                if idx < self.nr_batches_train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    l_train += loss.item()
                else:
                    l_test += loss.item()
                    
                if self.verbose:
                    pbar.update(1)

            avg_loss_train = l_train / self.nr_batches_train
            avg_loss_test = (l_test / self.nr_batches_test) if self.nr_batches_test > 0 else 0
            
            self.loss_train.append(avg_loss_train)
            self.loss_test.append(avg_loss_test)
            
            if self.verbose:
                pbar.set_postfix(TRAIN_LOSS=avg_loss_train, TEST_LOSS=avg_loss_test)
                
            if self.early_stop is not None and epoch > 0:
                if self.loss_train[epoch-1] < self.loss_train[epoch] + self.early_stop:
                    if self.verbose:
                        pbar.set_description(f'Early stop after epoch {(epoch+1):2}')
                    break
        
        pbar.close()
                
    def get_embedding(self, which: str = 'in'):
        """
        Returns fitted embeddings.
        
        Kwarg :which: 
            'in' to return first embedding, 
            'out' to return second embedding,
            'avg' to return their average
        """
        if which == 'in':
            return self.p2v.ivectors.weight.data
        elif which == 'out':
            return self.p2v.ovectors.weight.data
        elif which == 'avg':
            out = self.get_embedding('in')
            out = out.add(self.get_embedding('out')) / 2
            return out
        
    def plot_loss(self):
        """
        Visualize loss development
        """
        epochs = [i+1 for i in range(len(self.loss_train))]
        fig = go.Figure() # type: ignore
        fig.add_trace(go.Scatter( # type: ignore
            y=self.loss_train, 
            x=epochs,
            mode='lines+markers',
            name=f'Train loss ({self.nr_batches_train} batches)'
        ))
        fig.add_trace(go.Scatter( # type: ignore
            y=self.loss_test, 
            x=epochs,
            mode='lines+markers',
            name=f'Test loss  ({self.nr_batches_test} batches)'
        ))
        fig.show()


class ProductMapper(TSNE):
    """ A wrapper around TSNE
    
    Usage:
    >> mapper = ProductMapper(**tsne_kwargs)
    >> mapper.fit(embedding)
    >> mapper.plot()
    
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    Plotting method adjusted from https://github.com/sbstn-gbl/p2v-map
    """
    
    def __init__(self, **kwargs):
        super(ProductMapper, self).__init__(**kwargs)
        
    def plot(self):
        plot_data = go.Scatter( # type: ignore
            x = self.embedding_[:,0],
            y = self.embedding_[:,1],
            text = [f'Product {idx}' for idx in range(len(self.embedding_))],
            mode='markers',
            marker=dict(
                size=14,
                #color=dt['c'].values,
                colorscale='Jet',
                showscale=False
        ))

        plot_layout = go.Layout( # type: ignore
            width=800,
            height=600,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=4), # type: ignore
            hovermode='closest'
        )

        fig = go.Figure(data=plot_data, layout=plot_layout) # type: ignore
        fig.show()
