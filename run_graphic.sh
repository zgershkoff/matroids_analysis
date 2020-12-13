#!/bin/bash
#
# This script will create a classifier to predict if a matroid is graphic.
# To change which json file is analyzed, change the last argument.
python3 ./src/gaussiannb.py ./json/hr-sz13-rk08-results.json
