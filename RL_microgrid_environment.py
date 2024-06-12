import matplotlib.pyplot as plt
from RL_read_energy_data import *
from RL_read_solar_data import *
from RL_read_wind_data import *
from RL_read_grid_costs import *

timesteps = 35040

import random
import numpy as np
import pandas as pd

np.random.seed(0)

from pymgrid import Microgrid
from pymgrid.modules import (
    BatteryModule,
    LoadModule,
    RenewableModule,
    GridModule,
    GensetModule)
from pymgrid.forecast import OracleForecaster


def calculate_wind_power(wind_speeds, radius, power_coefficient, air_density=1.225):
    """
    Calculate average wind power generation in kilowatts based on wind speeds.

    Parameters:
        wind_speeds (np.array): Array of wind speeds in m/s.
        radius (float): Radius of the wind turbine rotor in meters.
        power_coefficient (float): Coefficient of performance of the turbine (typically between 0.35 and 0.45).
        air_density (float): Density of air in kg/m^3 (approximately 1.225 kg/m^3 at sea level).

    Returns:
        np.array: Power generated in kilowatts for each interval.
    """
    # Area of the rotor swept by the blades (πr^2)
    area = np.pi * (radius ** 2)

    # Calculate power in watts using the wind power formula: P = 0.5 * ρ * A * v^3 * Cp
    power_watts = 0.5 * air_density * area * (wind_speeds ** 3) * power_coefficient

    # Convert power from watts to kilowatts
    power_kilowatts = power_watts / 1000
    power_kilowattshour = power_kilowatts * 0.25

    return power_kilowattshour

def adjust_load_profile(load_profile, timesteps, factor=0.8):
    # Convert load profile to a NumPy array if it's not already one
    if not isinstance(load_profile, np.ndarray):
        load_profile = np.array(load_profile)

    # Apply the factor to each element in the load profile
    scaled_profile = load_profile * factor

    length = len(scaled_profile)

    if length > timesteps:
        # Keep only the last 'timesteps' values (considering warm up period)
        adjusted_profile = scaled_profile[-timesteps:]
    elif length < timesteps:
        # Extend the load profile if it's shorter than required
        # This example pads with the last value, but other strategies can be used
        padding = [scaled_profile[-1]] * (timesteps - length)
        adjusted_profile = np.concatenate((scaled_profile, padding))
    else:
        # If the load profile already matches the timesteps, use it as is
        adjusted_profile = scaled_profile

    return adjusted_profile

def create_microgrid(Energy_consumption, combined_df, df_buildings, steps=35040):
    load_modules = {}
    plot = []
    RES = []

    # Access the first row assuming it contains the grid information
    grid_info = combined_df.iloc[len(df_buildings)]  # Use iloc to access by position
    horizon = int(grid_info['Horizon'])

    for house_id, load_profile in Energy_consumption.items():
        # Create the load module
        processed_profile = adjust_load_profile(load_profile, steps)
        load_module = LoadModule(time_series=adjust_load_profile(load_profile, steps))
        plot.append(processed_profile[(steps // 52 )* 2 :(steps // 52 )* 3])
        # # Set the forecaster separately if the LoadModule class requires it
        load_module.set_forecaster(forecaster="oracle",
                                   forecast_horizon=horizon,
                                   forecaster_increase_uncertainty=False,
                                   forecaster_relative_noise=False)

        load_modules[f'load_{house_id}'] = load_module
        print("this is the length of the load module", len(load_module))

    # Filter out the rows where the Type is "Battery"
    battery_df = combined_df[combined_df["Type"] == "Battery"]

    # Group by the characteristics that define 'sameness' and sum relevant columns
    grouped_battery_df = battery_df.groupby(['Min_capaci', 'Max_capaci', 'Max_charge', 'Max_discha', 'Efficiency']).agg(
        {
            'Min_capaci': 'first',  # We only need the first entry as all are the same in the group
            'Max_capaci': 'first',
            'Max_charge': 'sum',  # Summing charge capacities
            'Max_discha': 'sum',  # Summing discharge capacities
            'Efficiency': 'first',  # Efficiency should be the same for grouped items
            'Battery_co': 'mean'  # Assuming you want the average cost cycle if they differ
        }).reset_index(drop=True)

    # Initialize a dictionary to store battery modules
    battery_modules = {}

    # Iterate over the grouped DataFrame
    for index, row in grouped_battery_df.iterrows():
        # Create a BatteryModule instance for each aggregated group
        battery_module = BatteryModule(
            min_capacity=row["Min_capaci"],
            max_capacity=row["Max_capaci"],
            max_charge=row["Max_charge"],
            max_discharge=row["Max_discha"],
            efficiency=row["Efficiency"],
            init_soc=random.uniform(0, 1),  # Initialize SOC randomly between 0 and 1
            battery_cost_cycle=row["Battery_co"]
        )

        # Store the battery module in the dictionary with its index as the key
        battery_modules[index] = battery_module

    solar_data_array = solar_data(timesteps, steps)
    # Convert the column to numeric, coercing errors to NaN
    df_buildings["TNO_dakopp_m2"] = pd.to_numeric(df_buildings["TNO_dakopp_m2"], errors='coerce')
    roof_partition = df_buildings["TNO_dakopp_m2"].sum() * 0.6
    # Print the calculated sum
    print("Horizontal roof partition calculated:", roof_partition)
    # problem is not all buildings have this
    efficiency_pv = 0.9
    solar_energy = roof_partition * solar_data_array * efficiency_pv

    RES.append((solar_energy[(steps // 52 )* 2 :(steps // 52 )* 3]))

    solar_energy = RenewableModule(time_series=(50*solar_energy))
    solar_energy.set_forecaster(forecaster="oracle",
                                             forecast_horizon=horizon,
                                             forecaster_increase_uncertainty=False,
                                             forecaster_relative_noise=False)

    # Filter out the rows where the Type is "Windturbine"
    windturbine_df = combined_df[combined_df["Type"] == "Windturbine"]
    # Dictionary to store wind modules
    wind_modules = {}

    # Iterate through the filtered DataFrame
    for index, row in windturbine_df.iterrows():
        # If wind data varies for each turbine, call here; otherwise, call outside the loop as shown
        wind_speed_array = wind_data(timesteps, steps)
        wind_data_array = calculate_wind_power(wind_speed_array, row["Radius_WT"], row["Power_coef"])

        # Create a RenewableModule instance for each turbine
        wind_module = RenewableModule(time_series=(50*wind_data_array))

        # Configure the forecaster settings for the module
        wind_module.set_forecaster(
            forecaster="oracle",
            forecast_horizon=horizon,  # Assuming the horizon is 4 weeks, 24 hours per day
            forecaster_increase_uncertainty=False,
            forecaster_relative_noise=False
        )

        # Store the module in the dictionary with a unique key
        wind_modules[f'windmod_{index}'] = ("windmodule", wind_module)

    RES.append(((wind_data_array)*len(windturbine_df))[(steps // 52 )* 2 :(steps // 52 )* 3])

    # Ensure that Epot and heat_pump_cop are defined
    Epot = grid_info['Geo_potent']
    print(Epot)# Example value, define as per actual data or requirements
    heat_pump_cop = 4.5  # Example coefficient of performance

    # Initialize total variables
    total_base_electrical_load = 0
    total_energy_output = 0

    # If `combined_df` is the DataFrame and you want to fill NaNs for the column 'TNO_grond_opp_m2'
    df_buildings['TNO_grond_opp_m2'].fillna(0, inplace=True)

    # Iterating over DataFrame rows properly
    for index, row in df_buildings.iterrows():
        # Directly access the column value in the row, which is now guaranteed not to be NaN
        amount_squaremeters = row['TNO_grond_opp_m2']
        residential_homes = row['BAG_aantal_verblijfsobjecten']

        # Do something with amount_squaremeters if needed
        print(amount_squaremeters)

        # Calculate base electrical load for current row
        base_electrical_load_gjoule = Epot * amount_squaremeters

        # Constant for conversion from GJ to kWh
        gj_to_kWh = 277.778

        # Calculate the kWh for the given joules
        base_electrical_load = base_electrical_load_gjoule * gj_to_kWh * 0.25

        # Calculate total energy output for current row
        energy_output = base_electrical_load * heat_pump_cop
        print(energy_output)

        # limit energy output with amount of residential homes * 20kW
        max_energy_output = residential_homes * 20 * heat_pump_cop * 0.25# Convert 20 kW to watts
        print("residential homes", residential_homes)
        print("max energy output", max_energy_output)

        if energy_output > max_energy_output:
            energy_output = max_energy_output

        # Accumulate the totals
        total_base_electrical_load += base_electrical_load
        total_energy_output += energy_output

    print("The total energy output is", total_energy_output)

    # Create a single GensetModule instance with the total calculated values
    total_heat_pump = GensetModule(
        running_min_production=0,  # Assuming no minimum production specified
        running_max_production=(total_energy_output*0.8)*1.5, #effiency,
        genset_cost=0.5,  #standard is 0.4 divided by 4
        co2_per_unit=0.0,
        cost_per_unit_co2=0.0,
        start_up_time=2,
        wind_down_time=2,
        provided_energy_name='Heatpump'
    )

    max_import = grid_info['Max_import']
    max_export = grid_info['Max_export']
    co2_price = grid_info['CO2_price']

    # Now you can use these values in your application
    print("Max Import:", max_import)
    print("Max Export:", max_export)
    print("CO2 Price:", co2_price)

    import_array = interpolate_import_costs(timesteps, steps)

    RES.append(import_array[(steps // 52 )* 2 :(steps // 52 )* 3])
    # export_array = interpolate_export_costs(timesteps) # https://www.zonnepanelen-info.nl/blog/stappenplan-stroom-terugleveren-aan-het-net/ per kwh

    #import, export, Co2
    grid_ts = [0.2,0.1,co2_price] * np.ones((steps, 3))
    grid_ts[:, 0] = import_array
    grid_ts[:, 1] = import_array # just
    grid_ts[:, 0] = np.where(grid_ts[:, 0] < 0, 0, grid_ts[:, 0])
    grid_ts[:, 1] = np.where(grid_ts[:, 1] < 0, 0, grid_ts[:, 1])

    grid = GridModule(max_import=max_import,
                      max_export=max_export,
                      time_series=grid_ts)

    grid.set_forecaster(forecaster="oracle",
                                   forecast_horizon=horizon,
                                   forecaster_increase_uncertainty=False,
                                   forecaster_relative_noise=False)

    modules = [grid,
               ('Heatpump', total_heat_pump),
               ('solar_energy', solar_energy)
               ]

    combined_modules = (modules + list(load_modules.values()) + list(battery_modules.values())
                        + list(wind_modules.values()))
    microgrid = Microgrid(combined_modules)

    # print all input data
    plt.figure(figsize=(15, 10))

    # Create the first axes
    ax1 = plt.gca()  # Gets the current axis (creates if necessary)

    # Plot all datasets except the last on the first y-axis
    for i, sublist in enumerate(RES[:-1]):
        ax1.plot(sublist, label=f'Dataset {i + 1}')

    # Set labels for the first y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy (kWh)')
    ax1.set_title('Energy Generation and Market Prices')

    # Create a second y-axis
    ax2 = ax1.twinx()  # Creates a second y-axis sharing the same x-axis

    # Plot the last dataset on the second y-axis
    ax2.plot(RES[-1], label=f'Dataset {len(RES)} (Euros)', color='r')  # 'r' is for red, or choose any color

    # Set labels for the second y-axis
    ax2.set_ylabel('Price (Euros)')

    # Set the legend
    # Getting labels and handles for both axes and combining them for one legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2)

    # Add grid
    ax1.grid(True)

    # Show the plot
    # plt.show()
    return microgrid





