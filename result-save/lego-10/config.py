import os
from inputs.env import Environment
import math

################### DATA SET SELECTION ###################
# env_name = '30-60-90'
# env_name = '30-60-90+equilateral'
# env_name = '30-60-90+rectangle'
# env_name = "polyiamond-3"
# env_name = "45-45-90+rectangle"
# env_name = "labyrinth"
# env_name = "lego-test"
# env_name = "lego-test-2"
# env_name = "lego-test-2.1"
# env_name = "lego-test-2.2"
# env_name = "lego-test-2.3"
# env_name = "lego-test-3"
# env_name = "lego-test-3.1"
# env_name = "lego-test-3.2"
# env_name = "lego-test-4"
# env_name = "lego-test-4.1"
# env_name = "lego-test-4.2"
# env_name = "lego-test-5"
# env_name = "lego-test-6"
# env_name = "lego-test-7"
# env_name = "lego-test-7.1"
# env_name = "lego-test-8"
# env_name = "lego-test-8.1"
# env_name = "lego-test-8.2"
# env_name = "lego-test-8.3"
# env_name = "lego-test-9"
env_name = "lego-test-10"

################### DATA SET CONFIGURATION ###################
# format: "tile set name : (mirror all tiles?, size of the superset, number of training data)"
env_attribute_dict = {
    '30-60-90'             : (True,  9, 12000),
    '30-60-90+equilateral' : (True,  6, 1000),
    '30-60-90+rectangle'   : (True,  6, 7000),
    '45-45-90+rectangle'   : (False, 9, 7000),
    "labyrinth"            : (False, 9, 5000),
    "polyiamond-3"         : (False, 9, 2000),
    'lego-test'            : (True,  6, 1000),
    'lego-test-2'          : (True,  9, 7000),
    'lego-test-2.1'        : (True,  11,4000),
    'lego-test-2.2'        : (True,  15,7000),
    'lego-test-2.3'        : (True,  2, 50),
    'lego-test-3'          : (True,  7, 5000),
    'lego-test-3.1'        : (True,  7, 4000),
    'lego-test-3.2'        : (False, 7, 1000),
    'lego-test-4'          : (True,  9, 2000),
    'lego-test-4.1'        : (True,  11,7000),
    'lego-test-4.2'        : (True,  7, 2000),
    'lego-test-5'          : (True,  7, 2000),
    'lego-test-6'          : (True,  11,7000),
    'lego-test-7'          : (True,  11,100),
    'lego-test-7.1'        : (True,  10,100),
    'lego-test-8'          : (True,  11,7000),
    'lego-test-8.1'        : (True,  11,2000),
    'lego-test-8.2'        : (True,  11,2000),
    'lego-test-8.3'        : (True,  11,5000),
    'lego-test-9'          : (True,  11,500),
    'lego-test-10'         : (True,  11,5000),
}

symmetry_tiles, complete_graph_size, number_of_data = env_attribute_dict[env_name]
env_location = os.path.join('.', 'data', env_name)
environment = Environment(env_location, symmetry_tiles=symmetry_tiles)
# SET YOUR DATASET PATH HERE
dataset_path = os.path.join('./dataset', f"{env_name}-ring{complete_graph_size}-{number_of_data}")

################### CREATING DATA ###################
shape_size_lower_bound=0.4
shape_size_upper_bound=0.6
max_vertices=20
validation_data_proportion=0.2

################### NETWORK PARAMETERS ###################
network_depth = 20
network_width = 32


################### TRAINING ###################
new_training = True
rand_seed = 2
batch_size = 1
learning_rate = 1e-3
training_epoch = 60
save_model_per_epoch = 1

COLLISION_WEIGHT    = 0.02 #1/math.log(1+1e-1)
ALIGN_LENGTH_WEIGHT = 0.02
AVG_AREA_WEIGHT     = 1
NODE_WEIGHT         = 5

################### DEBUGGING ###################
debug_data_num = 10
debug_base_folder = ".."
experiment_id = 4011 # unique ID to identify an experiment

#################### TESTING ##################
output_tree_search_layout = False
silhouette_path = "/home/edwardhui/data/silhouette/selected_v2"
network_path = f"./pre-trained_models/{env_name}.pth"
