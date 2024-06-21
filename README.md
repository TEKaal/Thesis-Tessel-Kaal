# Thesis

This thesis explores the integrating of various renewable energy sources into microgrids. The study seeks to minimize the burden on the main grid and minimize costs. Additionally, location-based optimization through hierarchical clustering is showcased in order to set the stage for further advancements in this framework to expand the complexity of the state spaces, and to minimize transmission losses further. The posed research question(s) are as follows: 

*Research question: How can Distributed Energy Resources be deployed and managed within an electrified microgrid to achieve a balance between energy consumption and production in order to minimize the burden on the central grid given fluctuating demand?*

To effectively address the main research question, it's divided into several sub-questions. This division allows for a targeted and in-depth exploration of each component of the main issue. The sub-questions are as follows:
1. Which modeling aspects need to be considered when designing a microgrid and its energy management system, and how are these reflected in the existing literature?
2. How can reinforcement learning be utilized to effectively place and manage all identified modeling aspects in a microgrid while maintaining grid stability and reliability?
3. How can the location-based optimization of a microgrid be effectively visualized to ensure it creates value for all stakeholders?



The repository includes scripts for data collection, data processing, analysis, and visualization to draw conclusions about microgrid optimization utililzing a DQN. 

# Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/TEKaal/Thesis-Tessel-Kaal.git
```

Navigate to the project directory:
```bash
cd Thesis-Tessel-Kaal
```

Install the required dependencies using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```

#Usage
Detailed explanations for each script and module are provided within the respective files. 

Change the input and output paths, and use the flags to pass the correct parameters. Run the main script that orchestrates the entire process, use:
```bash
python ./main.py
```

This script will train and evaluate the agent and output a training curve and evaluation graph, as well as all the data in CSVs. 

In the repository https://github.com/TEKaal/Support_Scripts_Thesis/ supporting scripts are provided to visualize the graphs in insightfull ways. 
