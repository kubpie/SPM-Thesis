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


file1 = "grakn17-output.txt"
file2 = 'grakn16-output.txt'
file3 = 'grakn15-output.txt'
file4 = 'graknKGCN-output.txt'
file5 = 'graknKGCN-output2.txt'
file6 = 'graknKGCN-output3.txt'
file7 = 'graknKGCN-output-newschema.txt'

var17,u17 = unique_vars(file1)
var16,u16 = unique_vars(file2)
var15,u15 = unique_vars(file3)

var, u = unique_vars(file4)
var2, u2 = unique_vars(file5)
var3,u3 = unique_vars(file6)
var4,u4 = unique_vars(file7)