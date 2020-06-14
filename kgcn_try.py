# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:50:54 2020

@author: kubap
"""

import copy
import inspect
import time
import pickle
import networkx
import matplotlib.pyplot as plt
import sys
#sys.path.append('/kglib')

from pathlib import Path
from random import shuffle

from grakn.client import GraknClient

from kglib.kgcn.pipeline.pipeline import pipeline
from kglib.utils.graph.iterate import multidigraph_data_iterator
from kglib.utils.graph.query.query_graph import QueryGraph
from kglib.utils.graph.thing.queries_to_graph import build_graph_from_queries

from kglib.utils.grakn.type.type import get_thing_types, get_role_types #missing in vehicle

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))