'''
    This file aims at giving paths that can be used to navigate through the repository (looking for data, etc.).
'''

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
DATA_DIR = os.path.join(ROOT_DIR, "data")
SRC_DIR = os.path.join(ROOT_DIR, "src")
