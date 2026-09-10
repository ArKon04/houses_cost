import numpy as np


def transform_target(y):
    return np.log1p(y)



def inverse_transform_target(y):
    return np.expm1(y)