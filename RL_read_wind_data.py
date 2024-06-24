import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def wind_data(steps):
    file_path = 'Wind power/Wind-KNMI'
    # Load the data using regex to handle multiple spaces and correctly interpret columns
    df = pd.read_csv(file_path, sep='\s{2,}', parse_dates=['DTG'], engine='python')

    # Setting 'DTG' as the index for easier resampling
    df.set_index('DTG', inplace=True)

    # Create a date range for the expected output including all 15-minute intervals
    start_time = '2022-01-01 00:15:00'  # Ensuring we start from the specified time
    date_range = pd.date_range(start=start_time, end=df.index.max(), freq='15min')

    # Reindex the wind series to the new date range with 15-minute intervals, keeping existing data
    wind_series_resampled = df['FF_SENSOR_10'].reindex(date_range)

    # Manually interpolate only the necessary 15-minute and 45-minute marks
    for dt in wind_series_resampled.index:
        if dt.minute % 30 == 15:  # checks if the minute is 15 or 45
            before = dt - pd.Timedelta(minutes=15)
            after = dt + pd.Timedelta(minutes=15)
            # Interpolate only if both surrounding data points exist
            if pd.notna(wind_series_resampled.get(before)) and pd.notna(wind_series_resampled.get(after)):
                wind_series_resampled[dt] = (wind_series_resampled[before] + wind_series_resampled[after]) / 2

    if pd.isna(wind_series_resampled.iloc[0]):
        wind_series_resampled.iloc[0] = 5.49
    # Print the resampled series to verify correct handling

    # Fill remaining NaNs by taking the value from the previous day at the same time
    wind_series_resampled.ffill(limit=96*7, inplace=True)

    print(wind_series_resampled.head(20))

    # Convert interpolated data to numpy array (if required, else can return as Series or DataFrame)
    wind_data_array = wind_series_resampled.to_numpy()

    return wind_data_array[:steps]