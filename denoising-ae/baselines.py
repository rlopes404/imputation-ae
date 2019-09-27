#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:21:02 2019

@author: ramon
"""

import numpy as np
np.random.seed(0)
import pandas as pd


def mean_imputation(df):
    mean_values = df.mean()