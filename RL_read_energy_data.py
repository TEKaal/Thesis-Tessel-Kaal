import numpy as np
import csv
import pandas as pd
import os

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

def process_energy_consumption_files(folder_path, building_list, timeinterval):
    energy_consumption_data = {}
    time_interval_seconds = timeinterval * 60  # 15 minutes in seconds

    for file_name in os.listdir(folder_path):
        if file_name.startswith('Building_'):
            building_id = file_name.split('_')[1].split('.')[0]
            if int(building_id) in building_list:
                consumption_data = []
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    next(reader, None)  # Skip header to ignore column titles
                    for row in reader:
                        total_energy_joules = float(row[1])  # Energy in Joules
                        # Convert Joules to Watts
                        total_power_watts = total_energy_joules / 3600000
                        consumption_data.append(total_power_watts)
                energy_consumption_data[building_id] = consumption_data

    return energy_consumption_data


def load_buildings_from_file(casestudy_file_name):
    # Load the specified columns from the file
    df = pd.read_csv(casestudy_file_name, usecols=['TARGET_FID_12_13', 'identificatie', 'POINT_X', 'POINT_Y', 'TNO_p_dak_horizontaal', 'TNO_dakopp_m2', 'BAG_aantal_verblijfsobjecten', 'BAG_oppervlakte'])

    # Extract columns
    ids = df['TARGET_FID_12_13'].astype(int)  # Convert TARGET_FID_12_13 to integers
    identification = df['identificatie'].astype(int)
    coordinates = df[['POINT_X', 'POINT_Y']].to_numpy()  # Combine POINT_X and POINT_Y into a coordinates array
    horizontal_roof = df['TNO_p_dak_horizontaal'].astype(float)  # Convert TNO_p_dak_horizontaal to float
    roof_surface = df['TNO_dakopp_m2'].astype(float)
    df["TYPE"] = "Load"
    type = df["TYPE"]
    # print(df)

    return df, coordinates, horizontal_roof, ids, type


def load_DERs_from_file(scenario_file_name, ids_buildings):
    # Load all columns from the file
    df = pd.read_csv(scenario_file_name)

    # Convert 'OBJECTID2' to numeric, adjust for non-numeric values, and ensure it's an integer
    df['ID_DER'] = pd.to_numeric(df['ID_DER'], errors='coerce').fillna(0).astype(int)

    # Adjust the 'ids' by adding the length of 'ids_buildings' to make them consequential
    offset = len(ids_buildings)
    print("this is the offset" , offset)
    df['ID_DER'] += offset

    # Extract adjusted ids and other columns
    ids = df['ID_DER']
    coordinates = df[['POINT_X', 'POINT_Y']].to_numpy()  # Combine POINT_X and POINT_Y into a coordinates array
    type_der = df['Type']


    return df, coordinates, ids, type_der

def concatenate_and_combine_columns(buildings, ders):
    # Rename the ID and type columns in both DataFrames to 'ID' and 'Type' respectively
    buildings_renamed = buildings.rename(columns={"TARGET_FID_12_13": "ID", "TYPE": "Type"})
    ders_renamed = ders.rename(columns={"ID_DER": "ID", "Type": "Type"})  # Ensure 'Type' column is in ders if not, adjust accordingly

    # Concatenate the two DataFrames
    combined_dataframe = pd.concat([buildings_renamed, ders_renamed], ignore_index=True)

    print(combined_dataframe.head())
    return combined_dataframe

