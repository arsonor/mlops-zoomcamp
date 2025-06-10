from pathlib import Path
import pickle
import pandas as pd
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import flow, task
from prefect.context import get_run_context


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


@task
def read_dataframe(year, month, color="yellow"):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    print(f"📥 Loaded {len(df)} raw records from {color} trip data {year}-{month:02d}")

    if color == "green":
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df = df.rename(columns={"lpep_pickup_datetime": "pickup_datetime"})
    elif color == "yellow":
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df = df.rename(columns={"tpep_pickup_datetime": "pickup_datetime"})
    else:
        raise ValueError("Invalid taxi color. Use 'green' or 'yellow'.")

    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    print(f"✅ Remaining records after preparation: {len(df)}")

    return df

@task
def create_X(df, dv=None):
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

@task
def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        print(f"🔍 Intercept: {lr.intercept_}")

        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        models_folder = Path("models")
        models_folder.mkdir(exist_ok=True)

        with open(models_folder / "preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(models_folder / "preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")

        return run.info.run_id

@flow
def main(execution_date: datetime = None, color: str = "green"):
    if execution_date is None:
        context = get_run_context()
        execution_date = context.flow_run.expected_start_time

    train_date = execution_date - relativedelta(months=2)
    val_date = execution_date - relativedelta(months=1)

    df_train = read_dataframe(train_date.year, train_date.month, color)
    df_val = read_dataframe(val_date.year, val_date.month, color)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"✅ MLflow run_id: {run_id}")


if __name__ == "__main__":
    main(execution_date=datetime(2023, 5, 1), color="yellow")