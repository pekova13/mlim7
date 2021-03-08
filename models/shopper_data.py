
from __future__ import annotations

import csv
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


class ShopperData(Dict[int, List[int]]):
    """
    A dictionary-like representation of weekly (keys) purchase data for a shopper.
    """
    shopper: int = -1

    def __init__(self, shopper: int):
        self.shopper = shopper

    def add(self, week: int, values: List[int]):
        """
        Add purchase data (`values`) for a given `week`. 
        Used by the `ShopperDataStreamer` to fill `ShopperData`.
        """
        self[week] = values

    def get(self, week) -> List[int]:
        """
        Get purchase data for a given `week`.
        """
        return self[week]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} for shopper {self.shopper}: {super().__repr__()}'


class ShopperDataWriter:
    """
    A utility to transform shopper data from a pandas DataFrame into a sparse textual representation
    which can be then streamed using `ShopperDataWriter`.

    Data is saved as a csv-file, with parameters saved in the first row
    `(max_shopper, max_week, max_value, target)` and data starting in the third row
    `(shopper, week, *values)`.

    Usage:
    >>> ShopperDataWriter.fit(df=df, target='target_variable').write(path='data.csv')
    """
    data: Dict[int, Dict[int, List[int]]]
    max_shopper: int = 0
    max_week: int = 0
    max_value: int = 0
    _fitted: bool = False
    
    def fit(self, df: pd.DataFrame, target: str) -> ShopperDataWriter:
        """
        Transform shopper data from a pandas DataFrame into a sparse textual representation
        which can be then streamed using `ShopperDataWriter`. Don't forget to `.write()` it to disc.
        """
        self.max_shopper = max(df['shopper'])
        self.max_week = max(df['week'])
        self.max_value = max(df[target])
        self.target = target

        self.data = {}
        
        row_iterator = df.iterrows()
        
        for _, row in tqdm(row_iterator, total=len(df)):
            
            self.data.setdefault(
                row['shopper'], {}
            ).setdefault(
                row['week'], []
            ).append(
                row[target]
            )
        
        self._fitted = True
        return self
    
    def write(self, path: str) -> None:
        """
        Write transformed shopper data to disc.
        """
        assert self._fitted

        with open(path, 'w') as f:
            writer = csv.writer(f)

            writer.writerow([self.max_shopper, self.max_week, self.max_value, self.target])
            writer.writerow([])

            for shopper in tqdm(range(self.max_shopper+1), total=self.max_shopper+1):
                for week in range(self.max_week+1):
                    try:
                        row = [shopper, week] + self.data[shopper][week]
                    except KeyError:
                        # there seem to be some missing values in the dataset, e.g. for shopper=56503 week=0
                        print(f'missing values for {shopper=} {week=}')
                        row = [shopper, week]
                    writer.writerow(row)


class ShopperDataStreamer:
    """
    A memory-efficient streamer for data stored in the ShopperData type.

    Usage:
    >>> streamer = ShopperDataStreamer('shopper_data.csv')
    >>> shopper_data: ShopperData
    >>> for shopper_data in streamer:
            pass
    >>> streamer.reset() # to re-use streamer
    >>> streamer.close() # close connection once streamer not needed
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.open()
    
    def open(self) -> None:
        """
        Open connection to the file. Called automatically at initiation.
        """
        self.file = open(self.path, 'r')
        self.reader = csv.reader(self.file)

        parms = next(self.reader)
        self.max_shopper = int(parms[0])
        self.max_week = int(parms[1])
        self.max_value = int(parms[2])
        self.target = parms[3]
        _ = next(self.reader)

    def reset(self) -> None:
        """
        Reset data streamer.
        """
        self.close()
        self.open()

    def close(self) -> None:
        """
        Close connection to the file. After closing, streamer can't be used.
        """
        self.file.close()

    def __next__(self) -> ShopperData:
        """
        Returns `ShopperData` for the next shopper.

        Raises StopIteration once rows have been exhausted. 
        If needed, call `.reset()` to repeat iteration.
        """
        try:
            shopper, week, *values = next(self.reader)

            shopper_data = ShopperData(shopper=int(shopper))
            shopper_data.add(
                week=int(week), 
                values=[int(value) for value in values]
            )

            # query next row as long as it still belongs to the same shopper
            while int(week) < self.max_week:
                _, week, *values = next(self.reader)
                shopper_data.add(
                    week=int(week),
                    values=[int(value) for value in values]
                )

            return shopper_data

        except StopIteration:
            raise StopIteration

    def __iter__(self):
        return self

    def __repr__(self):
        return (
            f'<{self.__class__.__name__} with {self.max_shopper+1} shoppers, '
            f'{self.max_week+1} weeks, and {self.max_value+1} {self.target}s>'
        )


if __name__ == '__main__':

    baskets_df = pd.read_parquet('data/baskets.parquet')[0:1_000_000]
    coupons_df = pd.read_parquet('data/coupons.parquet')[0:1_000_000]

    ShopperDataWriter().fit(df=baskets_df, target='product').write('baskets.csv')
    ShopperDataWriter().fit(df=coupons_df, target='product').write('coupon_products.csv')
    ShopperDataWriter().fit(df=coupons_df, target='discount').write('coupon_values.csv')

    prices: pd.DataFrame = baskets_df.groupby('product')['price'].agg('median').to_dict()
    prices.to_csv('prices.csv', index=False) # TODO fix this
