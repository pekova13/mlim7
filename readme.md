
### Pipeline

1. Transform data into the ShopperData format:
````
source venv/bin/activate
python models/shopper_data.py
````
- Transforms data from parquet files to csv files, where: First row contains summary statistics, second row is empty, consequent rows have a format `(shopper, week, *values)`.
- `baskets.csv` (values=products purchased), `coupon_products.csv` (products for which coupons were assigned), `coupon_values.csv` (assigned coupon values)
- This is slow!

2. Train the model:
````
source venv/bin/activate
python models/model.py
````