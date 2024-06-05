# microgrid_env = MicrogridEnv(env_config=env_config, modules=microgrid.modules)

# trained_agent = RL_helpfunctions.train_agent(microgrid_env, 1000)  # Train with 1000 episodes
# RL_helpfunctions.evaluate_agent(microgrid_env,trained_agent)

# print(microgrid_env)
# print(microgrid_env.get_action(action=1))

# print("Env observation_spec: \n", microgrid_env.observation_space)
# print("Env action_spec: \n", microgrid_env.action_space)
# print("Env reward_spec: \n", microgrid_env.reward_range)



# OR OTHER OPTION WITH THE TRANSITION VIA THE LIBRARY
# agent = Agent(gamma=0.99, epsilon=1.0, batch_size=128, nb_actions=5,
#               eps_end=0.01, input_dims=[10], lr=0.0001)

# nb_episode = 15
#
# for i in range(nb_episode):
#     score = 0
#     done = False
#     observation = microgrid_env.reset()
#     while not done:
#         action = agent.choose_action(observation)
#         observation_, reward, done, info = microgrid_env.step(action)
#         score -= reward
#         agent.store_transition(observation, action, reward, observation_, done)
#         agent.train()
#         observation = observation_
#     print(f"- episode : {i} | score : {score}")

# env = DiscreteEnvironment_modified
# agent = QLearner(env.observation_space.n, env.action_space.n)
# # RL interaction loop
# act_loop(env, agent, NUM_EPISODES=nr_timestamps) #NUM OF EPISODES

class MicrogridEnv(ContinuousMicrogridEnv):
    """Custom Environment that follows gym interface, accepting a pre-initialized microgrid"""
    def __init__(self, env_config, modules):
        super().__init__()
        self.seed = env_config.get('seed', 42)
        np.random.seed(self.seed)
        energy_consumption = env_config['energy_consumption']
        combined_df = env_config['combined_df']
        df_buildings = env_config['df_buildings']

        # Create the microgrid using the setup function
        self.microgrid = create_microgrid(energy_consumption, combined_df, df_buildings)

        # Assuming that the distance matrix and other necessary components are part of the microgrid object
        self.distance_matrix = env_config.get('normalized_distance_matrix')
        self.microgrid.control_dict = self.microgrid.controllable

        # Define action space
        # Define action space dimensions for each type of component
        # energy_source_actions = 3  # e.g., off, on, adjust output

        battery_actions = 3  # e.g., do nothing, charge, discharge
        grid_actions = 3  # e.g., none, import max, export max

        self.microgrid.action_space = spaces.MultiDiscrete([battery_actions] * len(self.microgrid.modules.battery) +
                                                 [grid_actions])

        # Mapping van de sources en welke load zij hun energie aan het geven zijn
        self.microgrid.observation_space = self.define_observation_space()

    def define_observation_space(self):
        # Access the log columns
        log_columns = self.microgrid.log.columns
        print(log_columns)
        # Assuming each variable has a range that needs to be specified
        # For simplicity, using a dictionary to define possible ranges
        # You might need to adjust these based on actual microgrid specs
        variable_ranges = {
            "P_pv": (0, 100),  # Example: PV power generation from 0 to 100 kW
            "SOC": (0, 100),  # State of charge for batteries from 0% to 100%
            # Add other variables as needed
        }

        # Create bounds based on available log columns
        high = []
        low = []
        for column in log_columns:
            if column in variable_ranges:
                low_value, high_value = variable_ranges[column]
                low.append(low_value)
                high.append(high_value)
            else:
                # Default range if variable is not predefined
                low.append(-100)  # Placeholder value, adjust as needed
                high.append(100)  # Placeholder value, adjust as needed

        # Convert lists to numpy arrays with dtype float32 for gym compatibility
        high = np.array(high, dtype=np.float32)
        low = np.array(low, dtype=np.float32)

        # Define the observation space using gym's Box space
        self.microgrid.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.microgrid.reset()  # Reset the microgrid state
        return self.get_initial_state()

    def step(self, action):
        # UPDATE THE MICROGRID
        action_dict = self.get_action(action)
        self.microgrid.run(action_dict, False)

        # COMPUTE NEW STATE AND REWARD
        self.state = self.transition()
        self.reward = self.get_reward()
        self.done = self.microgrid.done
        self.info = {}
        self.round += 1

        return self.state, self.reward, self.done, self.info

    def render(self, mode='human'):
        print(f"Current grid usage: {self.microgrid.modules.grid.current_usage()} kW")
        print(f"Current transmission losses: {self.calculate_transmission_loss()} kWh")

    def is_operation_done(self):
        return self.current_time_step >= 24

    def calculate_reward(self):
        # Example of penalizing grid usage and rewarding lower transmission losses
        grid_usage = self.microgrid.modules.grid.current_usage()  # This would need to be implemented
        transmission_loss = self.calculate_transmission_loss()  # This method also needs to be defined based on distance matrix

        # Adjust weights according to the importance of each factor
        penalty_for_grid_usage = -0.5 * grid_usage
        penalty_for_transmission_loss = -0.5 * transmission_loss
        return penalty_for_grid_usage + penalty_for_transmission_loss

    def calculate_transmission_loss(self):
        # Assuming you have a way to measure or estimate losses based on the distance matrix and current energy flows
        loss = 0
        loss_coefficient = 0.2
        for source_id, module in enumerate(self.microgrid.modules.energy_sources):
            for load_id, load in enumerate(self.microgrid.modules.loads):
                energy_transferred = self.current_energy_flow[source_id][load_id]  # Needs actual implementation
                distance = self.distance_matrix[source_id][load_id]
                loss += energy_transferred * distance * (
                    loss_coefficient)  # loss_coefficient to be defined based on your system's characteristics
        return loss

    def find_closest_source(self, load_id):
        distances = self.distance_matrix[:, load_id]
        closest_source_id = np.argmin(distances)
        return closest_source_id

    def get_action(self, action):
        control_dict = {}

        # Map each action integer to specific control strategies
        if action == 0:
            # Example: all batteries do nothing, no grid interaction
            pass
        elif action == 1:
            # Example: all batteries charge
            for i in range(len(self.microgrid.modules.battery)):
                control_dict[f'battery_{i}'] = {'operation': 'charge', 'amount': 'max'}
        elif action == 2:
            # Example: all batteries discharge
            for i in range(len(self.microgrid.modules.battery)):
                control_dict[f'battery_{i}'] = {'operation': 'discharge', 'amount': 'max'}
        # Add more actions as needed
        # add for the main grid

        return control_dict
    def get_initial_state(self):
        # print(self.microgrid.log.columns)
        initial_state = np.zeros(self.microgrid.observation_space.shape)
        for i, source in enumerate(self.microgrid.modules.wind_energy):
            print(source)
            initial_state[i] = source.current_renewable
        for i, battery in enumerate(self.microgrid.modules.battery, start=len(self.microgrid.modules.wind_energy)):
            initial_state[i] = battery.soc
        for i, load in enumerate(self.microgrid.modules.load,
                                 start=len(self.microgrid.modules.wind_energy) + len(self.microgrid.modules.battery)):
            initial_state[i] = load.current_load

    def get_next_state(self):
        # Calculate the next state based on microgrid responses to actions
        next_state = np.zeros(self.observation_space.shape)
        for i, source in enumerate(self.microgrid.energy_sources):
            next_state[i] = source.current_output
        for i, battery in enumerate(self.microgrid.batteries, start=len(self.microgrid.energy_sources)):
            next_state[i] = battery.soc
        for i, load in enumerate(self.microgrid.loads,
                                 start=len(self.microgrid.energy_sources) + len(self.microgrid.batteries)):
            next_state[i] = load.current_demand
        return next_state

#-------------------------------------------------------------------------
def add_transmission_costs(microgrid_env, distance_matrix, cost_per_distance_unit, id_matrix):
    # Function to calculate the transmission cost
    def calculate_transmission_cost(self, transfers):
        transmission_cost = 0
        energy_transfers_detail = []
        for transfer in transfers:
            source_index = self.module_index(transfer['source'])
            destination_index = self.module_index(transfer['destination'])
            distance = distance_matrix[source_index, destination_index]
            energy = transfer['amount']
            transmission_cost += cost_per_distance_unit * distance * energy
            energy_transfers_detail.append({
                'source': transfer['source'],
                'destination': transfer['destination'],
                'amount': energy,
                'cost': cost_per_distance_unit * distance * energy
            })
        return transmission_cost, energy_transfers_detail

    # Function to retrieve module index based on the module name or identifier
    def module_index(self, module_name):
        # Assuming the microgrid has a dictionary mapping module names to indices
        return self.module_name_to_index[module_name]

    # Extended step function to include transmission cost calculation
    def extended_step(self, action):
        obs, reward, done, info = super(type(self), self).step(action)  # Call original step method
        transfers = info.get('current_transfers', [])  # Assume this contains the transfer details
        transmission_cost, energy_transfers_detail = calculate_transmission_cost(self, transfers)
        # Adjust reward based on grid interactions
        for transfer in energy_transfers_detail:
            if 'grid' in transfer['source'] or 'grid' in transfer['destination']:
                reward -= abs(transfer['amount']) * 10  # POTENTIALLY CHANGE THIS
        reward -= transmission_cost
        info['energy_transfers'] = energy_transfers_detail
        return obs, reward, done, info

    # Bind the new methods to the microgrid object
    microgrid_env.calculate_transmission_cost = calculate_transmission_cost.__get__(microgrid_env)
    microgrid_env.module_index = module_index.__get__(microgrid_env)
    microgrid_env.step = extended_step.__get__(microgrid_env)  # Override the existing step method


    # Initialize a module name to index mapping if it's not already present
    microgrid_env.module_name_to_index = id_matrix




def simulate_MG(microgrid, iterations):
    microgrid.controllable
    microgrid.reset()

    # env = DiscreteEnvironment_modified.Environment()#
    # for module in microgrid.modules.items():
    #     print(f'{module}\n')

    # print(microgrid.state_series())



    # control_list = []
    # for i in range(len(microgrid.modules.battery)):
    #     control_list.append(battery_discharges[i])
    # print(control_list)

    control = {"grid": [grid_import],
               "battery": [battery_discharges[0],
                           battery_discharges[1],
                           battery_discharges[2]]}
    # ,
    #                            battery_discharges[3],
    #                            battery_discharges[4],
    #                            battery_discharges[5],
    #                            battery_discharges[6],
    #                            battery_discharges[7],
    #                            battery_discharges[8],
    #                            battery_discharges[9]]
    #still make this automated. --> does control_list work

    print("this is the log columns", microgrid.log.columns)

    # Execute control
    obs, reward, done, info = microgrid.run(control, normalized=False)
    print('OBS', obs)
    print('REWARD', reward)
    print('DONE', done)
    print('INFO', info)

    for module_type in ['load', 'solar_energy', 'wind_energy', 'battery', 'grid']:
        for module in getattr(microgrid.modules, module_type):
            print(f"{module_type} module step: {module.current_step}")

    for _ in range(iterations):
        microgrid.run(microgrid.sample_action(strict_bound=True))

    # for module_type in ['load', 'solar_energy', 'wind_energy', 'battery', 'grid']:
    #     for module in getattr(microgrid.modules, module_type):
    #         print(f"{module_type} module step: {module.current_step}")

    # print(microgrid.fixed)
    # print(microgrid.flex)
    # print(microgrid.controllable)

    columns_to_plot = [('load', 0, 'load_current'),
                       ('load', 0, 'load_met'),
                       ('wind_energy', 0, 'renewable_used'),
                       ('grid', 0, 'grid_import')]

    soc_column = [('battery', 0, 'soc')]  # SOC data to be plotted on the secondary y-axis

    # Create plot
    fig, ax1 = plt.subplots()

    # Plot the primary data on the primary y-axis
    microgrid.log[columns_to_plot].droplevel(axis=1, level=1).plot(ax=ax1)
    ax1.set_ylabel('Watt per timestep')  # Label for primary y-axis
    ax1.set_xlabel('Timesteps in 15 minutes')  # Set the x-axis label

    # Create a secondary y-axis for SOC
    ax2 = ax1.twinx()
    microgrid.log[soc_column].droplevel(axis=1, level=1).plot(ax=ax2, color='r', legend=False)
    ax2.set_ylabel('State of Charge (%)')  # Label for secondary y-axis

    # Adding legends manually if needed
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Show plot
    plt.title('Microgrid Energy Metrics and SOC')
    plt.show()


import gym
import numpy as np
from pymgrid.envs.discrete.discrete import DiscreteMicrogridEnv

class CustomDiscreteMicrogridEnv(DiscreteMicrogridEnv):
    def __init__(self, microgrid, buildings_df):
        # Make sure to initialize the superclass with necessary parameters
        super().__init__(microgrid)
        self.buildings_df = buildings_df

    @classmethod
    def from_microgrid(cls, microgrid, buildings_df):
        # Get the environment setup from the base class
        base_env = super(CustomDiscreteMicrogridEnv, cls).from_microgrid(microgrid)

        # Manually create an instance of the custom environment with all necessary parameters
        env = cls(microgrid, buildings_df)

        # Optionally, copy any additional state from base_env to env if needed
        # This step is necessary only if the base class `from_microgrid` does specific initializations
        # For example:
        # env.some_property = base_env.some_property

        return env

    def step(self, action):
        # Perform the original step action using the parent class
        obs, reward, done, info = super().step(action)

        # Custom reward adjustments based on your logic
        grid_usage = info.get('grid_usage', 0)
        total_load = sum([info.get(f'load_{i + 1}_absorbed', 0) for i in range(len(self.buildings_df))])
        percent_grid_usage = grid_usage / total_load if total_load > 0 else 0
        grid_penalty = -2 * percent_grid_usage + 1
        grid_penalty = np.clip(grid_penalty, -1, 1)

        reward += grid_penalty

        return obs, reward, done, info

    # bins_per_dimension = {
    #     ('load', 'load_current'): 2,
    #     ('load', 'load_forecast_0'): 2,
    #     ('load', 'load_forecast_1'): 2,
    #     # ('load', 'load_forecast_2'): 2,
    #     # ('load', 'load_forecast_3'): 2,
    #     ('renewable', 'renewable_current'): 2,
    #     ('renewable', 'renewable_forecast_0'): 2,
    #     ('renewable', 'renewable_forecast_1'): 2,
    #     # ('renewable', 'renewable_forecast_2'): 2,
    #     # ('renewable', 'renewable_forecast_3'): 2,
    #     'grid_status_current': 2,
    #     'import_price_current': 2,
    #     'export_price_current': 2,
    #     # 'import_price_forecast_0': 2,
    #     # 'export_price_forecast_0': 2,
    #     # 'grid_status_forecast_0': 2,
    #     # 'import_price_forecast_1': 2,
    #     # 'export_price_forecast_1': 2,
    #     # 'grid_status_forecast_1': 2,
    #     # 'import_price_forecast_2': 2,
    #     # 'export_price_forecast_2': 2,
    #     # 'grid_status_forecast_2': 2,
    #     # 'import_price_forecast_3': 2,
    #     # 'export_price_forecast_3': 2,
    #     # 'grid_status_forecast_3': 2,
    #     'soc':2,
    #     'current_charge':2
    # }
    def discharge_batteries_and_update_net_load(initial_net_load, battery_modules):
        """
        Discharges batteries sequentially based on the available net load and updates the net load accordingly.

        Parameters:
        - initial_net_load: The initial net load before discharging batteries.
        - battery_modules: A list of battery modules within the microgrid.

        Returns:
        - updated_net_load: The net load after sequentially discharging all batteries.
        - discharges: A list of discharge amounts for each battery.
        """
        updated_net_load = initial_net_load
        discharges = []

        for battery in battery_modules:
            discharge_amount = min(-1 * updated_net_load, battery.max_production)
            updated_net_load += discharge_amount
            discharges.append(discharge_amount)

        return updated_net_load, discharges

    solar_energy = microgrid.modules.solar_energy.item().current_renewable
    net_load = solar_energy

    wind_energy_list = []
    for i in range(0, len(microgrid.modules.windmodule)):
        wind_energy_list.append(microgrid.modules.windmodule[i].current_renewable)
        net_load += wind_energy_list[i]

    # wind_energy = microgrid.modules.wind_energy.item().current_renewable
    # dit is nu niet wind nog maar hoe die hernoemen.
    # Calculate net load

    total_load_per_timestep = []

    # print(microgrid.modules.load[1].current_load)
    for i in range(0, len(microgrid.modules.load)):
        total_load_per_timestep.append(-1.0 * microgrid.modules.load[i].current_load)

    print(net_load)
    for i in range(0, len(microgrid.modules.load)):
        net_load += total_load_per_timestep[i]

    # print("this is the net load:", net_load)
    if net_load > 0:
        net_load = 0.0

    updated_net_load, battery_discharges = discharge_batteries_and_update_net_load(net_load, microgrid.modules.battery)
    grid_import = min(-1 * updated_net_load, microgrid.modules.grid.item().max_production)
    control = {"grid": [grid_import],
                            "battery": [battery_discharges[0],
                           battery_discharges[1],
                           battery_discharges[2]]}

    microgrid.microgrid.controllable(control)

    # bins_per_dimension = {
    #     ('load', 'load_current'): 10,
    #     ('load', 'load_forecast_0'): 10,
    #     ('load', 'load_forecast_1'): 10,
    #     # ('load', 'load_forecast_2'): 2,
    #     # ('load', 'load_forecast_3'): 2,
    #     ('renewable', 'renewable_current'): 10,
    #     ('renewable', 'renewable_forecast_0'): 10,
    #     ('renewable', 'renewable_forecast_1'): 10,
    #     # ('renewable', 'renewable_forecast_2'): 2,
    #     # ('renewable', 'renewable_forecast_3'): 2,
    #     'grid_status_current': 2,
    #     'import_price_current': 10,
    #     # 'export_price_current': 10,
    #     # 'import_price_forecast_0': 2,
    #     # 'export_price_forecast_0': 2,
    #     # 'grid_status_forecast_0': 2,
    #     # 'import_price_forecast_1': 2,
    #     # 'export_price_forecast_1': 2,
    #     # 'grid_status_forecast_1': 2,
    #     # 'import_price_forecast_2': 2,
    #     # 'export_price_forecast_2': 2,
    #     # 'grid_status_forecast_2': 2,
    #     # 'import_price_forecast_3': 2,
    #     # 'export_price_forecast_3': 2,
    #     # 'grid_status_forecast_3': 2,
    #     'soc':10,
    #     'current_charge':5
    # }