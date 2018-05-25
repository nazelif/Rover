#!/bin/sh
python dem_generator.py
python elifs_dstar.py grid_result.txt
python visualize.py
