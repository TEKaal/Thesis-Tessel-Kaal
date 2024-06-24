import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def solar_data(steps):
    file_path = 'Solar/Solar-KNMI'
    # Load the data using regex to handle multiple spaces and correctly interpret columns
    df = pd.read_csv(file_path, sep='\s{2,}', parse_dates=['DTG'], engine='python')

    # Setting 'DTG' as the index for easier resampling
    df.set_index('DTG', inplace=True)

    # Create a date range for the expected output including all 15-minute intervals
    start_time = '2022-01-01 00:15:00'  # Ensuring we start from the specified time
    date_range = pd.date_range(start=start_time, end=df.index.max(), freq='15min')

    # Reindex the wind series to the new date range with 15-minute intervals, keeping existing data
    solar_series_resampled = df['Q_GLOB_10'].reindex(date_range)

    # Manually interpolate only the necessary 15-minute and 45-minute marks
    for dt in solar_series_resampled.index:
        if dt.minute % 30 == 15:  # checks if the minute is 15 or 45
            before = dt - pd.Timedelta(minutes=15)
            after = dt + pd.Timedelta(minutes=15)
            # Interpolate only if both surrounding data points exist
            if pd.notna(solar_series_resampled.get(before)) and pd.notna(solar_series_resampled.get(after)):
                solar_series_resampled[dt] = (solar_series_resampled[before] + solar_series_resampled[after]) / 2

    # Convert Watts to Kilowatts
    solar_series_resampled = solar_series_resampled / 1000
    solar_series_resampled_kwh = solar_series_resampled * 0.25

    # Print the resampled series to verify correct handling
    if pd.isna(solar_series_resampled_kwh.iloc[0]):
        solar_series_resampled_kwh.iloc[0] = 0.0

    # Fill remaining NaNs by taking the value from the previous day at the same time
    solar_series_resampled_kwh.ffill(limit=96*7, inplace=True)


    # Convert interpolated data to numpy array (if required, else can return as Series or DataFrame)
    solar_data_array = solar_series_resampled_kwh.to_numpy()
    print(solar_data_array)

    return solar_data_array[:steps]