
import pickle
import pandas as pd
import numpy as np
import argparse
import os

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def prediction(df):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch scoring for taxi trip duration prediction')
    parser.add_argument('--year', type=int, required=True, help='Year for the data (e.g., 2023)')
    parser.add_argument('--month', type=int, required=True, help='Month for the data (e.g., 4)')
    
    args = parser.parse_args()
    year = args.year
    month = args.month
    
    print(f"Processing data for {year}-{month:02d}")
    
    # Construct the URL for the specific year and month
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    print(f"Reading data from: {url}")
    
    # Read the data
    df = read_data(url)

    # Make predictions
    y_pred = prediction(df)

    # Calculate and print statistics
    mean_duration = np.mean(y_pred)
    std_dev = np.std(y_pred)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of predictions: {len(y_pred)}")
    print(f"Mean predicted duration: {mean_duration:.4f} minutes")
    print(f"Standard deviation of predicted duration: {std_dev:.4f} minutes")

    # Create artificial ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Create results dataframe with ride_id and predictions
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    print(f"\nResults dataframe shape: {df_result.shape}")

    # Save as parquet file
    output_file = f'predictions_{year}_{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Check the file size
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)

    print(f"\nOutput file saved: {output_file}")
    print(f"File size: {file_size} bytes ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()



