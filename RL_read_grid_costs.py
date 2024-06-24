import numpy as np
import pandas as pd

def interpolate_import_costs(steps):
    file_path = r'C:\Users\tessel.kaal\Documents\GitHub\Thesis\Grid\importcosts_grid_filtered.csv'  # Update with the actual path to your file

    # Load the data
    df = pd.read_csv(file_path, sep=',', parse_dates=['datumtijd'], index_col='datumtijd')

    # Create a date range that covers the desired interval with 15-minute steps
    start_time = '2022-01-01 00:15:00'
    date_range = pd.date_range(start=start_time, end=df.index.max(), freq='15min')

    # Reindex the DataFrame to this new date range, introducing NaN values for missing timestamps
    import_costs_resampled = df['Inkoop prijs per kWh'].reindex(date_range)

    # Manually interpolate only the necessary 15-minute and 45-minute marks
    for dt in import_costs_resampled.index:
        if dt.minute % 30 == 15:  # checks if the minute is 15 or 45
            before = dt - pd.Timedelta(minutes=15)
            after = dt + pd.Timedelta(minutes=15)
            # Interpolate only if both surrounding data points exist
            if pd.notna(import_costs_resampled.get(before)) and pd.notna(import_costs_resampled.get(after)):
                import_costs_resampled[dt] = (import_costs_resampled[before] + import_costs_resampled[after]) / 2

    # Handle the first data point specifically if it's NaN
    if pd.isna(import_costs_resampled.iloc[0]):
        import_costs_resampled.iloc[0] = 0.0  # Assuming a default or estimated value if missing

    # Fill remaining NaN values by taking the value from the previous day at the same time
    import_costs_resampled.ffill(limit=96 * 7,
                                 inplace=True)

    import_costs_array = import_costs_resampled.to_numpy()
    import_costs_array = np.divide(import_costs_array,4) # IN WATT

    # print(import_costs_array)

    return import_costs_array[:steps]
def interpolate_export_costs(timesteps):
    return np.random.uniform(0.05, 0.15, timesteps)
