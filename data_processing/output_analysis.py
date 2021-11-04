# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:26:56 2020

@author: kubap
"""

import numpy as np

import os

def unique_vars(file):
    path = os.getcwd()+"\\"
    varlist = []
    with open(path+file, 'r') as readfile:
        for line in readfile.readlines():
            varset = line.split('; ')
            varset[0] = varset[0].replace('{','')
            varlist.append(varset[:-1])
    unique = np.unique(varlist).tolist()
    return varlist, unique


#file1 = "grakn17-output.txt"
#file2 = 'grakn16-output.txt'
#file3 = 'grakn15-output.txt'
#file4 = 'graknKGCN-output.txt'
#file5 = 'graknKGCN-output2.txt'
#file6 = 'graknKGCN-output3.txt'
file1 = 'graknKGCN-output-all.txt'
file2 = 'graknKGCN-output-simple.txt'

var1,u1 = unique_vars(file1)
var2,u2 = unique_vars(file2)
