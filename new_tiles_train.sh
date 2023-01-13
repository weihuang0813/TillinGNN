#!/bin/bash
#generate new .pkl file
#generate new dataset
#training~

python gen_complete_super_graph.py
python gen_data.py
python network_train.py

exit 0
