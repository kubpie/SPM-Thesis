# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:50:54 2020

@author: kubap
"""

import copy
import inspect
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from grakn.client import GraknClient
from kglib.kgcn.pipeline.pipeline import pipeline
from kglib.utils.graph.iterate import multidigraph_data_iterator
from kglib.utils.graph.query.query_graph import QueryGraph
#from kglib.utils.graph.thing.queries_to_graph import build_graph_from_queries
from kglib.utils.grakn.type.type import get_thing_types, get_role_types #missing in vehicle
from kglib.utils.grakn.object.thing import build_thing
from kglib.utils.graph.thing.concept_dict_to_graph import concept_dict_to_graph

import tensorflow as tf
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.compat.v1.Session(config=config)
### Test tf for GPU acceleration
# TODO: Issues with GPU acceleration
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.reset_default_graph()
import warnings
from functools import reduce

KEYSPACE = "ssp_schema_kgcn"
URI = "localhost:48555"

import os
from data_prep import LoadData, FeatDuct
path = os.getcwd()+'\data\\'
raw_data = LoadData(path)
data = FeatDuct(raw_data, Input_Only = True) #leave only model input
data_complete = pd.read_csv(path+"data_complete.csv")

# DATA SELECTION FOR GRAKN TESTING
data = pd.concat([data.iloc[0:10,:],data.iloc[440:446,:],data.iloc[9020:9026,:]])
#data = pd.concat([data.iloc[0:3,:],data.iloc[440:443,:]])
#data = data.iloc[9020:9022,:]
# Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to exist
PREEXISTS = 0
# Candidates are neither present in the input nor in the solution, they are negative samples
CANDIDATE = 1
# Elements to infer are the graph elements whose existence we want to predict to be true, they are positive samples
TO_INFER = 2

# Categorical Attribute types and the values of their categories
ses = ['Winter', 'Spring', 'Summer', 'Autumn']
locations = []
for ssp in data_complete['profile']:
    season = next((s for s in ses if s in ssp), False)
    location = ssp.replace(season, '')[:-1]
    location = location.replace(' ', '-')
    locations.append(location)
loc = np.unique(locations)
# Categorical Attributes and lists of their values
CATEGORICAL_ATTRIBUTES = {'season': ses,
                          'location': loc.tolist()}

                          #'duct_type': ["NotDuct","SLD","DC"]}
                          
# Continuous Attribute types and their min and max values
CONTINUOUS_ATTRIBUTES = {'depth': (0, 1500), 
                         'num_rays': (500, 15000), 
                         'slope': (-2, 2), 
                         'bottom_type': (1,2),
                         'length': (0, 44000),
                         'SSP_value':(1463.486641,1539.630391),
                         'grad': (-0.290954924,0.040374179),
                         'number_of_ducts': (0,2)}

TYPES_TO_IGNORE = ['candidate-convergence', 'scenario_id', 'probability_exists', 'probability_nonexists', 'probability_preexists']
ROLES_TO_IGNORE = ['candidate_resolution', 'candidate_scenario']

# The learner should see candidate relations the same as the ground truth relations, so adjust these candidates to
# look like their ground truth counterparts

TYPES_AND_ROLES_TO_OBFUSCATE = {'candidate-convergence': 'convergence',
                                'candidate_resolution': 'minimum_resolution',
                                'candidate_scenario': 'converged_scenario'}

def concept_dict_from_concept_map(concept_map, tx):
    """
    Given a concept map, build a dictionary of the variables present and the concepts they refer to, locally storing any
    information required about those concepts.

    Args:
        concept_map: A dict of Concepts provided by Grakn keyed by query variables

    Returns:
        A dictionary of concepts keyed by query variables
    """
    return {variable: build_thing(grakn_concept) for variable, grakn_concept in concept_map.map().items()}


def combine_2_graphs(graph1, graph2):
    """
    Combine two graphs into one. Do this by recognising common nodes between the two.

    Args:
        graph1: Graph to compare
        graph2: Graph to compare

    Returns:
        Combined graph
    """

    for node, data in graph1.nodes(data=True):
        if graph2.has_node(node):
            data2 = graph2.nodes[node]
            if data2 != data:
                raise ValueError((f'Found non-matching node properties for node {node} '
                                  f'between graphs {graph1} and {graph2}:\n'
                                  f'In graph {graph1}: {data}\n'
                                  f'In graph {graph2}: {data2}'))

    for sender, receiver, keys, data in graph1.edges(data=True, keys=True):
        if graph2.has_edge(sender, receiver, keys):
            data2 = graph2.edges[sender, receiver, keys]
            if data2 != data:
                raise ValueError((f'Found non-matching edge properties for edge {sender, receiver, keys} '
                                  f'between graphs {graph1} and {graph2}:\n'
                                  f'In graph {graph1}: {data}\n'
                                  f'In graph {graph2}: {data2}'))

    return nx.compose(graph1, graph2)


def combine_n_graphs(graphs_list):
    # TODO: Rewrite this to combine multiple sub-graphs from a single query => repeated variables!
    # instead of multiple queries
    """
    Combine N graphs into one. Do this by recognising common nodes between the two.

    Args:
        graphs_list: List of graphs to combine

    Returns:
        Combined graph
    """
    
    
    
    return reduce(lambda x, y: combine_2_graphs(x, y), graphs_list)


def build_graph_from_queries(query_sampler_variable_graph_tuples, grakn_transaction,
                             concept_dict_converter=concept_dict_to_graph, infer=True):
    """
    Builds a graph of Things, interconnected by roles (and *has*), from a set of queries and graphs representing those
    queries (variable graphs)of those queries, over a Grakn transaction

    Args:
        infer: whether to use Grakn's inference engine
        query_sampler_variable_graph_tuples: A list of tuples, each tuple containing a query, a sampling function,
            and a variable_graph
        grakn_transaction: A Grakn transaction
        concept_dict_converter: The function to use to convert from concept_dicts to a Grakn model. This could be
            a typical model or a mathematical model

    Returns:
        A networkx graph
    """

    query_concept_graphs = []

    for query, sampler, variable_graph in query_sampler_variable_graph_tuples:
    
        concept_maps = sampler(grakn_transaction.query(query, infer=infer))
        concept_dicts = [concept_dict_from_concept_map(concept_map, grakn_transaction) for concept_map in concept_maps]

        answer_concept_graphs = []
        for concept_dict in concept_dicts:
            try:
                answer_concept_graphs.append(concept_dict_converter(concept_dict, variable_graph))
            except ValueError as e:
                raise ValueError(str(e) + f'Encountered processing query:\n \"{query}\"')

        if len(answer_concept_graphs) > 1:
            query_concept_graph = combine_n_graphs(answer_concept_graphs) # !!! This is the combine function
            query_concept_graphs.append(query_concept_graph)
        else:
            if len(answer_concept_graphs) > 0:
                query_concept_graphs.append(answer_concept_graphs[0])
            else:
                warnings.warn(f'There were no results for query: \n\"{query}\"\nand so nothing will be added to the '
                              f'graph for this query')

    if len(query_concept_graphs) == 0:
        # Raise exception when none of the queries returned any results
        raise RuntimeError(f'The graph from queries: {[query_sampler_variable_graph_tuple[0] for query_sampler_variable_graph_tuple in query_sampler_variable_graph_tuples]}\n'
                           f'could not be created, since none of these queries returned results')

    concept_graph = combine_n_graphs(query_concept_graphs)
    return concept_graph

def create_concept_graphs(example_indices, grakn_session):
    """
    Builds an in-memory graph for each example, with an scenario_id as an anchor for each example subgraph.
    Args:
        example_indices: The values used to anchor the subgraph queries within the entire knowledge graph
        =>> SCENARIO_ID
        grakn_session: Grakn Session

    Returns:
        In-memory graphs of Grakn subgraphs
    """
    #for scnenario_id with open grakn session:
        #1. get_query_handles()
        #2. build_graph_from_queries()
        #3. obfuscate_labels() whatever it means
        #4. graph.name = scenario_idx
        #5. append graph to list of graphs
        
    graphs = []
    infer = True
    
    for scenario_idx in example_indices:
        print(f'Creating graph for example {scenario_idx}')
        graph_query_handles = get_query_handles(scenario_idx)
        with grakn_session.transaction().read() as tx:
            # Build a graph from the queries, samplers, and query graphs
            graph = build_graph_from_queries(graph_query_handles, tx, infer=infer)

        obfuscate_labels(graph, TYPES_AND_ROLES_TO_OBFUSCATE) #???

        graph.name = scenario_idx
        graphs.append(graph)

    return graphs

def obfuscate_labels(graph, types_and_roles_to_obfuscate):
    # Remove label leakage - change type labels that indicate candidates into non-candidates
    for data in multidigraph_data_iterator(graph):
        for label_to_obfuscate, with_label in types_and_roles_to_obfuscate.items():
            if data['type'] == label_to_obfuscate:
                data.update(type=with_label)
                break



def get_query_handles(scenario_idx):
    
    # Contains Schema-Specific queries that retrive sub-graphs from grakn!
    
    """
    Creates an iterable, each element containing a Graql query, a function to sample the answers, and a QueryGraph
    object which must be the Grakn graph representation of the query. This tuple is termed a "query_handle"

    Args:
        scenario_idx: A uniquely identifiable attribute value used to anchor the results of the queries to a specific
                    subgraph
        ===>>> SINGLE SCENARIO_ID

    Returns:
        query handles
    """
    # === Convergence ===
    conv, scn, ray, nray, src, dsrc, seg, dseg, l, s, srcp, bathy, bt, ssp, loc, ses,\
    sspval, dsspmax, speed, dssp, dct, ddct, gd, duct, nod = 'conv','scn','ray', 'nray',\
    'src', 'dsrc', 'seg', 'dseg','l','s','srcp','bathy','bt','ssp','loc','ses',\
    'sspval','dsspmax','speed','dssp','dct','ddct','gd','duct','nod'                        
    
    #dt,'dt',
    convergence_query = inspect.cleandoc(
        f'''match 
        $scn isa sound-propagation-scenario, has scenario_id {scenario_idx};'''
        '''$ray isa ray-input, has num_rays $nray; 
        $src isa source, has depth $dsrc; 
        $seg isa bottom-segment, has depth $dseg, has length $l, has slope $s;
        $conv(converged_scenario: $scn, minimum_resolution: $ray) isa convergence;
        $srcp(defined_by_src: $scn, define_src: $src) isa src-position;
        $bathy(defined_by_bathy: $scn, define_bathy: $seg) isa bathymetry, has bottom_type $bt;
        $ssp isa SSP-vec, has location $loc, has season $ses, has SSP_value $sspval, has depth $dsspmax;
        $dct isa duct, has depth $ddct, has grad $gd;
        $speed(defined_by_SSP: $scn, define_SSP: $ssp) isa sound-speed;
        $duct(find_channel: $ssp, channel_exists: $dct) isa SSP-channel, has number_of_ducts $nod; 
        $sspval has depth $dssp;
        {$dssp == $dsrc;} or {$dssp == $dseg;} or {$dssp == $ddct;} or {$dssp == $dsspmax;}; 
        get;'''
        ) # has duct_type $dt,
    
    '''
    get $scn, $sid, $ray, $nray, $conv,
    $src, $dsrc, $seg, $dseg, $l, $s, $srcp, $bathy, $bt,
    $ssp, $loc, $ses,  $dsspmax, $speed, $sspval, $dssp,
    $dct, $ddct, $gd, $dt, $duct, $nod;
    '''

    convergence_query_graph = (QueryGraph()
                             .add_vars([conv], TO_INFER)
                             .add_vars([scn, ray, nray, src, dsrc, seg, dseg, \
                                        l, s, srcp, bathy, bt, ssp, loc, ses, \
                                        sspval, dsspmax, speed, dssp, dct, ddct,\
                                         gd, duct, nod], PREEXISTS) #dt,
                             .add_has_edge(ray, nray, PREEXISTS)
                             .add_has_edge(src, dsrc, PREEXISTS)
                             .add_has_edge(seg, dseg, PREEXISTS)
                             .add_has_edge(seg, l, PREEXISTS)
                             .add_has_edge(seg, s, PREEXISTS)
                             .add_has_edge(ssp, loc, PREEXISTS)
                             .add_has_edge(ssp, ses, PREEXISTS)
                             .add_has_edge(ssp, sspval, PREEXISTS)
                             .add_has_edge(ssp, dsspmax, PREEXISTS)
                             .add_has_edge(dct, ddct, PREEXISTS)
                             #.add_has_edge(dct, dt, PREEXISTS)
                             .add_has_edge(dct, gd, PREEXISTS)
                             .add_has_edge(bathy, bt, PREEXISTS)
                             .add_has_edge(duct, nod, PREEXISTS)
                             .add_has_edge(sspval, dssp, PREEXISTS)
                             .add_role_edge(conv, scn, 'converged_scenario', TO_INFER) #TO_INFER VS CANDIDATE BELOW
                             .add_role_edge(conv, ray, 'minimum_resolution', TO_INFER)
                             .add_role_edge(srcp, scn, 'defined_by_src', PREEXISTS)
                             .add_role_edge(srcp, src, 'define_src', PREEXISTS)
                             .add_role_edge(bathy, scn, 'defined_by_bathy', PREEXISTS)
                             .add_role_edge(bathy, seg, 'define_bathy', PREEXISTS)
                             .add_role_edge(speed, scn, 'defined_by_SSP', PREEXISTS)
                             .add_role_edge(speed, ssp, 'define_SSP', PREEXISTS)
                             .add_role_edge(duct, ssp, 'find_channel', PREEXISTS)
                             .add_role_edge(duct, dct, 'channel_exists', PREEXISTS)
)

    # === Candidate Convergence ===
    candidate_convergence_query = inspect.cleandoc(f'''match
           $scn isa sound-propagation-scenario, has scenario_id {scenario_idx};
           $ray isa ray-input, has num_rays $nray;
           $conv(candidate_scenario: $scn, candidate_resolution: $ray) isa candidate-convergence; 
           get;''')
           

    candidate_convergence_query_graph = (QueryGraph()
                                       .add_vars([conv], CANDIDATE)
                                       .add_vars([scn, ray, nray], PREEXISTS)
                                       .add_has_edge(ray, nray, PREEXISTS)
                                       .add_role_edge(conv, scn, 'candidate_scenario', CANDIDATE)
                                       .add_role_edge(conv, ray, 'candidate_resolution', CANDIDATE))

    return [
        (convergence_query, lambda x: x, convergence_query_graph),
        (candidate_convergence_query, lambda x: x, candidate_convergence_query_graph)
        ]

def write_predictions_to_grakn(graphs, tx):
    """
    Take predictions from the ML model, and insert representations of those predictions back into the graph.

    Args:
        graphs: graphs containing the concepts, with their class predictions and class probabilities
        tx: Grakn write transaction to use

    Returns: None

    """
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            if data['prediction'] == 2:
                concept = data['concept']
                concept_type = concept.type_label
                if concept_type == 'convergence' or concept_type == 'candidate-convergence':
                    neighbours = graph.neighbors(node)

                    for neighbour in neighbours:
                        concept = graph.nodes[neighbour]['concept']
                        if concept.type_label == 'sound-propagation-scenario':
                            scenario = concept
                        else:
                            ray = concept

                    p = data['probabilities']
                    query = (f'match'
                             f'$scn id {scenario.id};'
                             f'$ray id {ray.id};'
                             #f'$kgcn isa kgcn;'
                             f'insert'
                             f'$conv(sound-propagation-scenario: $scn, ray-input: $ray) isa convergence,'
                             f'has probability_exists {p[2]:.3f},'
                             f'has probability_nonexists {p[1]:.3f},'  
                             f'has probability_preexists {p[0]:.3f};')
                    tx.query(query)
    tx.commit()

def convergence_example(data, num_graphs=100,
                      num_processing_steps_tr=5,
                      num_processing_steps_ge=5,
                      num_training_iterations=300,
                      keyspace=KEYSPACE, uri=URI):
    """
    Run the diagnosis example from start to finish, including traceably ingesting predictions back into Grakn

    Args:
        num_graphs: Number of graphs to use for training and testing combined
        num_processing_steps_tr: The number of message-passing steps for training
        num_processing_steps_ge: The number of message-passing steps for testing
        num_training_iterations: The number of training epochs
        keyspace: The name of the keyspace to retrieve example subgraphs from
        uri: The uri of the running Grakn instance

    Returns:
        Final accuracies for training and for testing
    """

    tr_ge_split = int(num_graphs*0.5) #training-test solit 50/50
    print(tr_ge_split)

    client = GraknClient(uri=uri)
    session = client.session(keyspace=keyspace)
    
    example_idx = data.index.tolist()
    #example_idx = example_idx[0:6]
    print(example_idx)
    graphs = create_concept_graphs(example_idx, session) 
    
    with session.transaction().read() as tx:
        # Change the terminology here onwards from thing -> node and role -> edge
        node_types = get_thing_types(tx)
        [node_types.remove(el) for el in TYPES_TO_IGNORE]

        edge_types = get_role_types(tx)
        [edge_types.remove(el) for el in ROLES_TO_IGNORE]
        print(f'Found node types: {node_types}')
        print(f'Found edge types: {edge_types}')
    
    ge_graphs, solveds_tr, solveds_ge = pipeline(graphs,
                                                 tr_ge_split,
                                                 node_types,
                                                 edge_types,
                                                 num_processing_steps_tr=num_processing_steps_tr,
                                                 num_processing_steps_ge=num_processing_steps_ge,
                                                 num_training_iterations=num_training_iterations,
                                                 continuous_attributes=CONTINUOUS_ATTRIBUTES,
                                                 categorical_attributes=CATEGORICAL_ATTRIBUTES,
                                                 output_dir=f"./events/{time.time()}/")

    #with session.transaction().write() as tx:
    #    write_predictions_to_grakn(ge_graphs, tx)
    
    session.close()
    client.close()
    
    return graphs#, ge_graphs, solveds_tr, solveds_ge

graphs =  convergence_example(data, num_graphs=len(data), #len(data)
                      num_processing_steps_tr=1, #5
                      num_processing_steps_ge=1, #5
                      num_training_iterations=100, #300
                      keyspace=KEYSPACE, uri=URI)

# TODO: shuffle data before feeding to grakn!!!
# TODO: Leave out validation set