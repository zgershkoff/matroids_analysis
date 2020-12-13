#!/bin/bash
#
# This script will create a model to predict the girth of a matroid.
# To change which json file is analyzed, change the last argument.
python3 ./src/svm.py ./json/hr-sz13-rk08-results.json
