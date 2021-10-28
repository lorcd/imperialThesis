#!/bin/bash

"""
Can use this to run M18-P. The argument after m18_run.py refers to the number of probabilistic layers we want to use out of the 18 overall"""

CUDA_VISIBLE_DEVICES=0 python3 -i m18_run.py 18 #arg must be 1, 5, 9, 13, 17, or 18
