# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:26:56 2020

@author: kubap
"""

g17 = open("grakn17-output.txt", "r")
g17_content = g17.readlines()

"""
from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    pass

with open(file, 'r') as readfile:
    f = readfile.readlines()
    for textblock in f:
        seen = OrderedCounter([line.strip() for line in textblock])
        print("\n".join([k for k,v in seen.items() if v == 1]))  
"""
import os
path = os.getcwd()+"\\"
file = "grakn17-output.txt"

varlist = []
with open(path+file, 'r') as readfile:
    for line in readfile.readlines():
        varset = line.split('; ')
        varset[0] = varset[0].replace('{','')
        varlist.append(varset[:-1])
        
unique = varlist[0][:]
for vset in varlist:
    for var in vset:
        if var not in unique:
            unique.append(var)
            

repeating = set(unique) - set(varlist[0][:])