import numpy as np
def create_directed_matrix(combined_df):
    # Initialize mappings from indices to IDs with type and count information
    index_to_id = {}
    type_count = {}  # To track the number of occurrences of each type

    # Initialize lists for sources and sinks based on Types
    sinks = []
    sources = []

    for index, row in combined_df.iterrows():
        module_type = row['Type']

        # Update type count and create specific identifiers
        if module_type == 1:
            # For loads, use 'identificatie' directly
            identifier = row['identificatie']
        else:
            # For other types, increment count and form a unique identifier
            if module_type not in type_count:
                type_count[module_type] = 1
            else:
                type_count[module_type] += 1

            # Define specific naming conventions for each type
            if module_type == 0:
                type_name = "grid"
            elif module_type == 2:
                type_name = "battery"
            elif module_type == 3:
                type_name = "windturbine"
            else:
                type_name = f"type_{module_type}"

            identifier = f"{type_name}_{type_count[module_type]}"
        index_to_id[index] = identifier

    for index, row in combined_df.iterrows():
        if row['Type'] in ["Grid", "Load", "Battery"]:  # Types 0 is grid , type 1 is  buildings, and 2 are batteries and they are both sinks and sources flow towards sinks and receive from sources
            sinks.append(index)
            sources.append(index)
        elif row['Type'] in ["Windturbine"]:  # Type 3 is a wind turbine and is thus directed towards sinks and thing sinks/sources, type 4 is tbd.
            sources.append(index)

    # Initialize the matrix
    num_points = len(combined_df)
    zero_matrix = np.zeros((num_points, num_points))

    # Populate the matrix, avoiding self-interaction
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                if i in sources and j in sinks:
                    zero_matrix[i][j] = 1

    return zero_matrix, index_to_id
def calculate_distance_matrix(combined_dataframe, matrix):
    num_points = combined_dataframe.shape[0]
    # Initialize a matrix to store the distances
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            # Calculate distance only if i is not equal to j
            if i != j:
                x1, y1 = combined_dataframe.loc[i, 'POINT_X'], combined_dataframe.loc[i, 'POINT_Y']
                x2, y2 = combined_dataframe.loc[j, 'POINT_X'], combined_dataframe.loc[j, 'POINT_Y']
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                distance_matrix[i, j] = distance

    connected_matrix = distance_matrix * matrix
    return connected_matrix, distance_matrix

def calculate_normalized_distance_matrix(combined_dataframe, matrix):
    num_points = combined_dataframe.shape[0]
    # Initialize a matrix to store the distances
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            if i != j:  # Calculate distance only if i is not equal to j
                x1, y1 = combined_dataframe.loc[i, 'POINT_X'], combined_dataframe.loc[i, 'POINT_Y']
                x2, y2 = combined_dataframe.loc[j, 'POINT_X'], combined_dataframe.loc[j, 'POINT_Y']
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                distance_matrix[i, j] = distance

    # Apply the connection matrix to mask out unconnected distances
    connected_matrix = distance_matrix * matrix

    # Normalize the connected_matrix
    max_distance = np.max(connected_matrix[connected_matrix > 0])  # find the maximum value greater than zero
    if max_distance > 0:  # avoid division by zero
        normalized_matrix = connected_matrix / max_distance
    else:
        normalized_matrix = connected_matrix  # if max_distance is zero, all connections are zero

        # Set disconnected paths to a high value
    normalized_matrix[normalized_matrix == 0] = 99999
    return normalized_matrix

