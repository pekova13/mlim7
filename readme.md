
# Pipeline

## Step 0. Setup environment
Create a virtual environment and install required packages:
````
python3 -m venv venv
pip -r requirements.txt
````
Activate the virtual environment:
````
source venv/bin/activate
````
Specify model parameters as desired in `steps/config.py`

## Step 1. Transform data into the ShopperData format
````
python steps/1_prepare_data.py
````
- Transforms data from parquet files to csv files, where: First row contains summary statistics, second row is empty, consequent rows have a format `(shopper, week, *values)`.
- `baskets.csv` (values=products purchased), `coupon_products.csv` (products for which coupons were assigned), `coupon_values.csv` (assigned coupon values)
- This is slow!

## Step 2. Train the model
````
python steps/2_train_model.py
````
## Step 2. Assign coupons
````
python steps/3_assign_coupons.py
````