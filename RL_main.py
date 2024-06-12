import argparse
import RL_helpfunctions
from RL_helpfunctionDQN import *
from RL_visualizegrid import *
from RL_microgrid_environment import *
from RL_read_energy_data import *
from RL_connection_matrix import *
from RL_custom_Env import *
import csv
from pymgrid.microgrid.trajectory.stochastic import FixedLengthStochasticTrajectory
from Cluster_algorithm import *
import optuna
import optuna.visualization as vis
import sys


def objective(trial):
    # Define the hyperparameters to tune
    dqn_episodes = trial.suggest_int('dqn_episodes', 1, 2)
    dqn_batch_size = trial.suggest_int('dqn_batch_size', 32, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    memory_size = trial.suggest_int('memory_size', 672*4, 672*8)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.9, 0.999)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    layer_size = trial.suggest_int('layer_size', 64, 256)

    # Use a fixed output folder for Optuna trials
    outputfolder = 'output_optuna'

    # Retrieve the environment from the study's user_attrs
    microgrid_env = trial.study.user_attrs['microgrid_env']

    # Set up the environment (assuming microgrid_env is already defined)
    # Ensure this is done before calling train and evaluate
    # microgrid_env = ...
    # microgrid_env = CustomMicrogridEnv.from_microgrid(microgrid)

    # Train the agent with the suggested hyperparameters
    agent = train_dqn_agent(microgrid_env, outputfolder, dqn_episodes, args.nr_steps, dqn_batch_size, learning_rate,
                            memory_size, num_layers, layer_size, epsilon_decay, gamma)

    # Evaluate the agent
    average_reward, _ = evaluate_dqn_agent(microgrid_env, outputfolder, agent, args.dqn_evaluation_steps, args.nr_steps)

    return average_reward


def save_arguments_to_csv(args, outputfolder):
    # Extract necessary arguments from args
    name_giving = input("Please enter the folder name: ")
    dqn_episodes = getattr(args, 'dqn_episodes', 800)  # Using defaults if not specified
    dqn_evaluation_steps = getattr(args, 'dqn_evaluation_steps', 200)

    # Create a directory named with the current date, run number, and other details
    current_date = datetime.now().strftime("%m%d")
    folder_name = f"{current_date}_{dqn_episodes}_{dqn_evaluation_steps}_{name_giving}"
    arguments_folder = os.path.join(outputfolder, folder_name)
    os.makedirs(arguments_folder, exist_ok=True)

    # Create the filename with a timestamp
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"arguments_{current_datetime}.csv"
    full_path = os.path.join(arguments_folder, csv_filename)

    # Write the arguments to the CSV file
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Argument', 'Value'])
        for arg, value in vars(args).items():
            writer.writerow([arg, value])

    return arguments_folder  # Return the new folder name

def main(args):
    # Define the time variables
    run_folder = save_arguments_to_csv(args, args.outputfolder)
    time_interval = args.time_interval

    # Load the case study and scenario files
    df_buildings, coordinates_buildings, horizontal_roof, ids_buildings, type_buildings = load_buildings_from_file(args.case_study_file)
    df_ders, coordinates_ders, ids_ders, type_der = load_DERs_from_file(args.scenario_file, ids_buildings)
    combined_df = concatenate_and_combine_columns(df_buildings, df_ders)

    # Load all the energy data
    Energy_consumption = process_energy_consumption_files(args.folder_path_loads, list(ids_buildings), time_interval)

    microgrid = create_microgrid(Energy_consumption, combined_df, df_buildings)
    print("Microgrid is created, now wrap in gym environment")
    print(microgrid)
    microgrid_env = CustomMicrogridEnv.from_microgrid(microgrid)

    # microgrid_env = CustomMicrogridEnv.from_scenario(microgrid_number=10)
    microgrid_env.trajectory_func = FixedLengthStochasticTrajectory(args.nr_steps)

    print("Initialising trajectory")

    microgrid_env.initial_step = 0
    microgrid_env.final_step = 26280

    trained_agent_DQN = train_dqn_agent(microgrid_env, run_folder, args.dqn_episodes, args.nr_steps, args.dqn_batch_size, args.learning_rate, args.memory_size,  args.num_layers, args.layers_size, args.epsilon_d, args.gamma)

    print('TRAINING DQL DONE')

    microgrid_env.initial_step = 26280
    microgrid_env.final_step = 35040

    evaluate_dqn_agent(microgrid_env, run_folder, trained_agent_DQN, args.dqn_evaluation_steps, args.nr_steps)

    return microgrid_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run microgrid simulation.')

    # File name containing the loads
    folder_path_loads = "Final loads"
    case_study_file = "Buildings and scenarios/CS1.csv"
    scenario_file = "Buildings and scenarios/Scenario1.csv"
    output_file = r"C:\Users\tessel.kaal\OneDrive - Accenture\Thesis\Output training model\VERSION 5\\VERSION 5.1"

    parser.add_argument('--outputfolder', type=str, default=output_file, help='Folder to save output files.')
    parser.add_argument('--folder_path_loads', type=str, default=folder_path_loads, help='Path to the folder containing load files.')
    parser.add_argument('--case_study_file', type=str, default=case_study_file, help='Path to the case study file.')
    parser.add_argument('--scenario_file', type=str, default=scenario_file, help='Path to the scenario file.')
    parser.add_argument('--nr_steps', type=int, default=96, help='Number of steps for the simulation.')
    parser.add_argument('--time_interval', type=int, default=15, help='Time interval in minutes.')
    parser.add_argument('--dqn_episodes', type=int, default=400, help='Number of episodes for DQN training.')
    parser.add_argument('--dqn_batch_size', type=int, default=64, help='Batch size for DQN training.')
    parser.add_argument('--dqn_evaluation_steps', type=int, default=200, help='Number of evaluation steps for DQN.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for DQN.')
    parser.add_argument('--memory_size', type=int, default=96*4, help='Memory allocation.')
    parser.add_argument('--num_layers', type=int, default=4, help='Neural network')
    parser.add_argument('--layers_size', type=int, default=64, help='Neural layer size')
    parser.add_argument('--epsilon_d', type=float, default=0.999, help='Memory allocation.')
    parser.add_argument('--gamma', type=float, default=1, help='Memory allocation.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna optimization.')

    args = parser.parse_args()

    # Initialize the environment in the main function
    microgrid_env = main(args)

    # # Use Optuna to find the best hyperparameters
    # study = optuna.create_study(direction='maximize')
    #
    # # Pass the environment and other static parameters to the study
    # study.set_user_attr('microgrid_env', microgrid_env)
    # study.set_user_attr('nr_steps', args.nr_steps)
    # study.set_user_attr('dqn_evaluation_steps', args.dqn_evaluation_steps)
    #
    # study.optimize(objective, n_trials=args.n_trials)
    #
    # # Print the best hyperparameters found
    # print("Best hyperparameters:", study.best_params)
    # print("Best value:", study.best_value)
    #
    # vis.plot_optimization_history(study)
    # vis.plot_param_importances(study)
    # vis.plot_slice(study)
