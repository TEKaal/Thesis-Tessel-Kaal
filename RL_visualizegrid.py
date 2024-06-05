import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_and_visualize_graph(combined_df, normalized_matrix):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for index, row in combined_df.iterrows():
        G.add_node(index, pos=(row['POINT_X'], row['POINT_Y']), node_type=row['Type'])

    # Add directed edges based on the normalized matrix
    num_nodes = normalized_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):  # Consider all node pairs (i, j)
            if normalized_matrix[i][j] > 0:  # Assuming 0 means no connection
                G.add_edge(i, j, weight=normalized_matrix[i][j])

    # Position dictionary for node plotting (optional customization for better visualization)
    pos = {node: (G.nodes[node]['pos']) for node in G.nodes()}

    # Prepare color map based on node types found in combined_df
    unique_types = set(combined_df['Type'].astype(int))  # Ensuring types are integers
    colors_palette = plt.cm.get_cmap('viridis', len(unique_types))  # Get a color map from matplotlib
    color_map = {type: colors_palette(i) for i, type in enumerate(unique_types)}

    # Node colors based on type
    colors = [G.nodes[node]['node_type'] for node in G.nodes()]
    node_colors = [color_map[type] for type in colors]

    # Draw the graph with arrows indicating the direction of edges
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='blue', arrowstyle='-|>', arrowsize=10, width=2, connectionstyle='arc3, rad=0.1')

    plt.title('Directed Microgrid Visualization')
    plt.axis('off')  # Turn off the axis numbers / grid
    plt.show()