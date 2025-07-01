import os
import pandas as pd
from datetime import datetime
from batch_q6 import save_data, read_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Step 1: Create the input dataframe
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)


# Step 2: Save to S3
input_file = os.getenv('INPUT_FILE_PATTERN').format(year=2023, month=1)
save_data(df_input, input_file)


# Step 3: Run batch_q6.py from Python
exit_code = os.system("python batch_q6.py 2023 1")
assert exit_code == 0, "batch_q6.py did not run successfully"


# Step 4: Read the output file
output_file = os.getenv('OUTPUT_FILE_PATTERN').format(year=2023, month=1)
df_result = read_data(output_file)


# Step 5: Check the result
print(df_result)
print("Sum of predicted durations:", df_result['predicted_duration'].sum())

