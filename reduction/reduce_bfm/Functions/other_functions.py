import numpy as np
import sys

def insw_vector(parameter):
    x = len(parameter)
    switch = np.zeros(x)
    for i in range(0,x):
        if parameter[i] > 0.0:
            switch[i] = 1.0

    return switch


def eTq_vector(temp, basetemp, q10):
    x = len(temp)
    eTq = np.zeros(x)
    for i in range(0,x):
        eTq[i] = np.exp(np.log(q10)*(temp[i]-basetemp)/basetemp)
    
    return eTq

def get_concentration_ratio(numerator, denominator, p_small):
    x = len(numerator)
    y = len(denominator)
    if x == y:
        concentration_ratio = np.zeros(x)
        for i in range(0,x):
            if numerator[i] > 0.0:
                concentration_ratio[i] = numerator[i]/(denominator[i] + p_small)
    else:
        sys.exit("Warning: Array lengths do not match. Cannot determine concentration ratios.")
    
    return concentration_ratio