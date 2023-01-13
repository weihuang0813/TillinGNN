#!/bin/bash
#already have .pkl file
#generate new dataset
#training~

python gen_data.py
python network_train.py

exit 0
