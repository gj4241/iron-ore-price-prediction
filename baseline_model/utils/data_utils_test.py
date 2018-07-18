# -*- coding: utf-8 -*-
"""
Test data utils

Created on Tue Jul 17 21:58:57 2018

@author: user
"""

import numpy as np

from data_utils import *

def test_series_to_supervised():
    print('test series_to_supervised()...')
    x = [0,1,2,3,4,5,6]
    print(x,'to supervised(1,1)')
    print(series_to_supervised(x,1,1))
    x = np.random.randn(20,2)
    print(x,'to supervised(4,2)')
    print(series_to_supervised(x,4,2))
    
if __name__ == "__main__":
    test_series_to_supervised()