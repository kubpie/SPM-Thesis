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
from pipeline_mod import pipeline
#from kglib.kgcn.pipeline.pipeline import pipeline
from kglib.utils.graph.iterate import multidigraph_data_iterator
from kglib.utils.graph.query.query_graph import QueryGraph
from kglib.utils.grakn.type.type import get_thing_types, get_role_types #missing in vehicle
#from kglib.utils.graph.thing.queries_to_graph import build_graph_from_queries
from kglib.utils.graph.thing.queries_to_graph import combine_2_graphs, combine_n_graphs, concept_dict_from_concept_map
from kglib.utils.grakn.object.thing import build_thing
from kglib.utils.graph.thing.concept_dict_to_graph import concept_dict_to_graph

from sklearn.model_selection import train_test_split

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
### Test tf for GPU acceleration
# TODO: Issues with GPU acceleration
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.reset_default_graph() #fix bugs with tensor of uknonw size

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #filter out mpl warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN) #filter out annoying messages about name format with ':'

import os

KEYSPACE =  "ssp_2class" #"ssp_schema_slope0"  #"sampled_ssp_schema_kgcn"
URI = "localhost:48555"

# Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to exist
PREEXISTS = 0
# Candidates are neither present in the input nor in the solution, they are negative samples
CANDIDATE = 1
# Elements to infer are the graph elements whose existence we want to predict to be true, they are positive samples
TO_INFER = 2

from pathlib import Path
from data_prep import LoadData, FeatDuct, UndersampleData
PATH = os.getcwd() #+'\data\\'
datapath = Path(PATH+"/data/")
ALLDATA = LoadData(datapath)
ALLDATA = FeatDuct(ALLDATA, Input_Only = True) #leave only model input
PROCESSED_DATA = pd.read_csv(str(datapath)+"/data_complete.csv")


# Categorical Attribute types and the values of their categories
ses = ['Winter', 'Spring', 'Summer', 'Autumn']
locations = []
for ssp in ALLDATA['profile']:
    season = next((s for s in ses if s in ssp), False)
    location = ssp.replace(season, '')[:-1]
    location = location.replace(' ', '-')
    locations.append(location)
loc = np.unique(locations).tolist()

# Categorical Attributes and lists of their values
CATEGORICAL_ATTRIBUTES = {'season': ses,
                          'location': loc}
                          #duct_type': ["NotDuct","SLD","DC"]}
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
        concept_dicts = [concept_dict_from_concept_map(concept_map) for concept_map in concept_maps]
        #TODO: Implement removal of NotDuct cases at NetworkX level instead of query workaround
        """
        #print(concept_dicts)
        notaduct = 0
        for cd in concept_dicts:
            print(cd)
            for variable, thing in cd.items(): #key, value
                if variable == 'gd' and thing.value == 0.0:#and '0.0' in value:
                    print(variable, thing.value)
                    #cd.pop('gd')
                    #cd.pop('dct')
                    #cd.pop('SSP-channel')
                    #cd.pop()
                    #val = 'grad' in cd['gd']
                    #print(val)
                    #if '0.0' in value:
                    #    print('asdasda')
        """
        answer_concept_graphs = []
        for concept_dict in concept_dicts:
            try:
                answer_concept_graphs.append(concept_dict_converter(concept_dict, variable_graph))
            except ValueError as e:
                raise ValueError(str(e) + f'Encountered processing query:\n \"{query}\"')

        if len(answer_concept_graphs) > 1:
            query_concept_graph = combine_n_graphs(answer_concept_graphs)
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
    #TODO: Remove NotDuct result from NetworkX graph completely: entity duct, attr grad 0, depth 0
     
    
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
        
    Outline:    
    For scnenario_id with open grakn session:
        0. check if the nx.graph for example doesn't exists already in the output directory
        if yes: load nx.graph from pickle file
        if no: 
            1. get_query_handles()
            2. build_graph_from_queries()
            3. obfuscate_labels() whatever it means
            4. graph.name = scenario_idx
            5. save ns.graph as pickle file
            6. append graph to list of graphs and return the list as func. output
            

    """
    
    graphs = []
    infer = True
    #savepath = f"./networkx/"
    savepath = PATH + "/networkx/"
    total = len(example_indices)
    
    not_duct_idx = []
    for idx, sld, dc in zip(range(len(PROCESSED_DATA)),PROCESSED_DATA['SLD_depth'],PROCESSED_DATA['DC_axis']):
        if np.isnan(sld) and np.isnan(dc):
            not_duct_idx.append(idx)
        
    for it, scenario_idx in enumerate(example_indices):
        graph_filename = f'graph_{scenario_idx}.gpickle'
        if not os.path.exists(str(savepath)+"/"+graph_filename):
            print(f'[{it+1}|{total}] Creating graph for example {scenario_idx}')
            graph_query_handles = get_query_handles(scenario_idx, not_duct_idx)
            #print(graph_query_handles)
            with grakn_session.transaction().read() as tx:
                # Build a graph from the queries, samplers, and query graphs
                graph = build_graph_from_queries(graph_query_handles, tx, infer=infer)
    
            obfuscate_labels(graph, TYPES_AND_ROLES_TO_OBFUSCATE)
    
            graph.name = scenario_idx
            nx.write_gpickle(graph, savepath+graph_filename)
        
        else:
            print(f'[{it+1}|{total}] NetworkX graph loaded {graph_filename}')
            graph = nx.read_gpickle(savepath+graph_filename)    
        
        graphs.append(graph)
        
        # new_graph = networkx.Graph(graph)
        # nx.draw(new_graph)
        # plt.show()
    return graphs

def obfuscate_labels(graph, types_and_roles_to_obfuscate):
    # Remove label leakage - change type labels that indicate candidates into non-candidates
    for data in multidigraph_data_iterator(graph):
        for label_to_obfuscate, with_label in types_and_roles_to_obfuscate.items():
            if data['type'] == label_to_obfuscate:
                data.update(type=with_label)
                break

def get_query_handles(scenario_idx, not_duct_idx):
        
    """
    Creates an iterable, each element containing a Graql query, a function to sample the answers, and a QueryGraph
    object which must be the Grakn graph representation of the query. This tuple is termed a "query_handle"
    Args:
        scenario_idx: A uniquely identifiable attribute value used to anchor the results of the queries to a specific subgraph
    Returns:
        query handles
    """
    # === Query variables ===
    conv, scn, ray, nray, src, dsrc, seg, dseg, l, s, srcp, bathy, bt, ssp, loc, ses,\
    sspval, dsspmax, speed, dssp, dct, ddct, gd, duct, nod = 'conv','scn','ray', 'nray',\
    'src', 'dsrc', 'seg', 'dseg','l','s','srcp','bathy','bt','ssp','loc','ses',\
    'sspval','dsspmax','speed','dssp','dct','ddct','gd','duct','nod'
    # dt, 'dt'
    
    
    # === Candidate Convergence ===
    candidate_convergence_query = inspect.cleandoc(f'''match
           $scn isa sound-propagation-scenario, has scenario_id {scenario_idx};'''
           '''$ray isa ray-input, has num_rays $nray;
           $conv(candidate_scenario: $scn, candidate_resolution: $ray) isa candidate-convergence; 
           get;''')    
          # {$nray == 500;} or {$nray == 1000;}; -> takes too much time to do it like that, better limit at data migration
 
    candidate_convergence_query_graph = (QueryGraph()
                                       .add_vars([conv], CANDIDATE)
                                       .add_vars([scn, ray, nray], PREEXISTS)
                                       .add_has_edge(ray, nray, PREEXISTS)
                                       .add_role_edge(conv, scn, 'candidate_scenario', CANDIDATE)
                                       .add_role_edge(conv, ray, 'candidate_resolution', CANDIDATE))

   
    
    if scenario_idx not in not_duct_idx:
        # === Convergence: SCN with ducts ===    
        convergence_query_full = inspect.cleandoc(
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
            )
        # has duct_type $dt,
        
        convergence_query_full_graph = (QueryGraph()
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
        return [
            (convergence_query_full, lambda x: x, convergence_query_full_graph),
            (candidate_convergence_query, lambda x: x, candidate_convergence_query_graph)
            ]
    
    
    else:        
        # === Convergence: SCN with\without ducts ===
        convergence_query_reduced = inspect.cleandoc(
                f'''match 
                $scn isa sound-propagation-scenario, has scenario_id {scenario_idx};'''
                '''$ray isa ray-input, has num_rays $nray; 
                $src isa source, has depth $dsrc; 
                $seg isa bottom-segment, has depth $dseg, has length $l, has slope $s;
                $conv(converged_scenario: $scn, minimum_resolution: $ray) isa convergence;
                $srcp(defined_by_src: $scn, define_src: $src) isa src-position;
                $bathy(defined_by_bathy: $scn, define_bathy: $seg) isa bathymetry, has bottom_type $bt;
                $ssp isa SSP-vec, has location $loc, has season $ses, has SSP_value $sspval, has depth $dsspmax;
                $speed(defined_by_SSP: $scn, define_SSP: $ssp) isa sound-speed;
                $sspval has depth $dssp;
                {$dssp == $dsrc;} or {$dssp == $dseg;} or {$dssp == $dsspmax;}; 
                get;'''
                )
            
        convergence_query_reduced_graph = (QueryGraph()
                                 .add_vars([conv], TO_INFER)
                                 .add_vars([scn, ray, nray, src, dsrc, seg, dseg, \
                                            l, s, srcp, bathy, bt, ssp, loc, ses, \
                                            sspval, dsspmax, speed, dssp], PREEXISTS)
                                 .add_has_edge(ray, nray, PREEXISTS)
                                 .add_has_edge(src, dsrc, PREEXISTS)
                                 .add_has_edge(seg, dseg, PREEXISTS)
                                 .add_has_edge(seg, l, PREEXISTS)
                                 .add_has_edge(seg, s, PREEXISTS)
                                 .add_has_edge(ssp, loc, PREEXISTS)
                                 .add_has_edge(ssp, ses, PREEXISTS)
                                 .add_has_edge(ssp, sspval, PREEXISTS)
                                 .add_has_edge(ssp, dsspmax, PREEXISTS)
                                 .add_has_edge(bathy, bt, PREEXISTS)                           
                                 .add_has_edge(sspval, dssp, PREEXISTS)
                                 .add_role_edge(conv, scn, 'converged_scenario', TO_INFER) #TO_INFER VS CANDIDATE BELOW
                                 .add_role_edge(conv, ray, 'minimum_resolution', TO_INFER)
                                 .add_role_edge(srcp, scn, 'defined_by_src', PREEXISTS)
                                 .add_role_edge(srcp, src, 'define_src', PREEXISTS)
                                 .add_role_edge(bathy, scn, 'defined_by_bathy', PREEXISTS)
                                 .add_role_edge(bathy, seg, 'define_bathy', PREEXISTS)
                                 .add_role_edge(speed, scn, 'defined_by_SSP', PREEXISTS)
                                 .add_role_edge(speed, ssp, 'define_SSP', PREEXISTS)
                                 )    
            

        return [
            (convergence_query_reduced, lambda x: x, convergence_query_reduced_graph),
            (candidate_convergence_query, lambda x: x, candidate_convergence_query_graph)
            ]

def write_predictions_to_grakn(graphs, tx, commit = True):
    """
    Take predictions from the ML model, and insert representations of those predictions back into the graph.

    Args:
        graphs: graphs containing the concepts, with their class predictions and class probabilities
        tx: Grakn write transaction to use

    Returns: None

    """
    
    #TODO: Revise these loops and see why nothing is being predicted as data['prediction']=2 (exists?)
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
                    query = (f'match '
                             f'$scn id {scenario.id}; '
                             f'$ray id {ray.id}; '
                             f'insert '
                             f'$conv(converged_scenario: $scn, minimum_resolution: $ray) isa convergence, '
                             f'has probability_exists {p[2]:.3f}, '
                             f'has probability_nonexists {p[1]:.3f}, '  
                             f'has probability_preexists {p[0]:.3f};')
                    print(query)
                    tx.query(query)
    if commit:
        tx.commit()

import re
def ubuntu_rand_fix():

    savepath = PATH + '/networkx/'
    graphfiles = [f for f in os.listdir(savepath) if os.path.isfile(os.path.join(savepath, f))]
    example_idx = []
    for gfile in graphfiles:
        idx = re.findall(r'\d+', gfile)[0]    
        example_idx.append(idx)
    return example_idx

def prepare_data(session, data, train_split, validation_split, ubuntu_fix = True):
    """
    Args:
        data: full dataset with sorted scenario_id's that will be used for querying grakn
        train_split: size of the training set; 
        validaton_split: size of the validaton set subtracted from the test set; 
    
        Test set is further split down into test and validation so that
        test_set size = (1-train_split)*(1-validation_split)
        so i.e. train_split = 0.7, validation_split=0.33 results in:
        70% training set, 20.1% test set, 9.9% validation set
    """
    seed = 123
    
    y = data.pop('num_rays').to_frame()
    X = data
    # divide whole dataset into stratified train\test 
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, shuffle = True, random_state = seed, test_size=1-train_split)
    #if validation_split > 0:
    #divide test dataset into stratified test\validation subsets
    #X_test, X_val, y_test, y_val = train_test_split(
    #X_test, y_test, stratify=y_test, shuffle = True, random_state = seed, test_size=validation_split)
    
    training_data = [X_train, y_train]
    testing_data = [X_test, y_test]
    #validation_data = [X_val, y_val]
    
    # data was split and shuffled while mainating original indices 
    # now the training and test set indices are merged once again
    # and will be split again inside the grakn pipeline until tr_ge_split, without shuffle
    
    num_tr_graphs = len(X_test) + len(X_train)   
    #num_val_graphs = len(X_val)
    example_idx_tr = X_train.index.tolist() + X_test.index.tolist() #training and test sets indices merged for training

    # rand in linux and windows generates different number in effect the data selected in windows is different than ubuntu
    if ubuntu_fix:
        example_idx_tr = ubuntu_rand_fix()
    #example_idx_val = X_val.index.tolist()
    tr_ge_split = int(num_tr_graphs * train_split)  # Define graph number split in train graphs[:tr_ge_split] and test graphs[tr_ge_split:] sets
    #val_ge_split = int(len(X_val)*(1-validation_split))
    print(f'\nCREATING {num_tr_graphs} TRAINING\TEST GRAPHS')
    train_graphs = create_concept_graphs(example_idx_tr, session)  # Create validation graphs in networkX
    #print(f'\nCREATING {num_val_graphs} VALIDATION GRAPHS')
    #val_graphs = create_concept_graphs(example_idx_val, session) # Create training graphs in networkX
    
    return  train_graphs, tr_ge_split, training_data, testing_data #, val_graphs,  val_ge_split

def go_train(train_graphs, tr_ge_split, save_fle, **kwargs):
    """
    Args:
           
    Parameters
    ----------
    train_graphs : networkx graphs obtained from grakn queries - the set contains both train and test graphs!
    tr_ge_split : int. value marking the number of training graphs in train_graphs
    save_fle : model filename to be saved as tf. checkpoin
    **kwargs : TYPE

    Returns:
    ge_graphs: Encoded in-memory graphs of Grakn concepts for generalisation
    solveds_tr: training fraction examples solved correctly
    solveds_ge: test/generalization fraction examples solved correctly

    """
    # Run the pipeline with prepared networkx graph
    ge_graphs, solveds_tr, solveds_ge, graphs_enc, input_graphs, target_graphs = pipeline(graphs = train_graphs,             
                                                tr_ge_split = tr_ge_split,                         
                                                do_test = False,
                                                save_fle = save_fle,
                                                reload_fle = "",
                                                **kwargs)
    
    training_evals= [solveds_tr, solveds_ge]   
    return ge_graphs, training_evals, graphs_enc, input_graphs, target_graphs
 
def go_test(val_graphs, val_ge_split, reload_fle, **kwargs):
    
    # opens session once again, if closed after training  
    client = GraknClient(uri=URI)
    session = client.session(keyspace=KEYSPACE)

    ge_graphs, solveds_tr, solveds_ge = pipeline(graphs = val_graphs,  # Run the pipeline with prepared graph
                                                 tr_ge_split = val_ge_split,
                                                 do_test = True,
                                                 save_fle = "",
                                                 reload_fle = reload_fle, 
                                                 **kwargs)
    
    with session.transaction().write() as tx:
        write_predictions_to_grakn(ge_graphs, tx)  # Write predictions to grakn with learned probabilities
    
    session.close()
    client.close()
    # Grakn session will be closed here due to write\insert query
    
    validation_evals = [solveds_tr, solveds_ge] 
    return ge_graphs, validation_evals

##### RUN THE PIPELINE  #####  

# DATA SELECTION FOR GRAKN TESTING
from data_analysis_lib import ClassImbalance
from data_prep import CreateSplits


#data = UndersampleData(ALLDATA, max_sample = 100)
#data = UndersampleData(data, max_sample = 30) #at 30 you got 507 nx graphs created, howeve with NotDuct at this point

# === 2 classes of 2000 sample 500/1000 ==== 
keyspace = "ssp_2class"
data_sparse2 = ALLDATA[(ALLDATA.loc[:,'num_rays'] == 500) | (ALLDATA.loc[:,'num_rays'] == 1000)]
data = UndersampleData(data_sparse2, max_sample = 2000)

# === 3 classes of 1020 samples: 500/6000/15000 ===== 
#keyspace = "ssp_3class"
#data_sparse3 = ALLDATA[(ALLDATA.loc[:,'num_rays'] == 500) | (ALLDATA.loc[:, 'num_rays'] == 1000)] #3classes  (ALLDATA.loc[:, 'num_rays'] == 1500) |
#data = UndersampleData(data_sparse3, max_sample = 1020)

class_population = ClassImbalance(data, plot = False)
print(class_population)


client = GraknClient(uri=URI)
session = client.session(keyspace=keyspace)

with session.transaction().read() as tx:
        # Change the terminology here onwards from thing -> node and role -> edge
        node_types = get_thing_types(tx)
        [node_types.remove(el) for el in TYPES_TO_IGNORE]
        edge_types = get_role_types(tx)
        [edge_types.remove(el) for el in ROLES_TO_IGNORE]
        print(f'Found node types: {node_types}')
        print(f'Found edge types: {edge_types}')   

train_graphs, tr_ge_split, training_data, testing_data = prepare_data(session, data, train_split=0.7, validation_split = 0.2, ubuntu_fix= False)
#, val_graphs,  val_ge_split

kgcn_vars = {
          'num_processing_steps_tr': 20, #10
          'num_processing_steps_ge': 20, #10
          'num_training_iterations': 5000, #100
          'learning_rate': 1e-4, #1e-3
          'latent_size': 16, #MLP param 16
          'num_layers': 3, #MLP param 3
          'clip': 50, #gradient clipping 5.0
          'weighted': False, #loss function modification
          'log_every_epochs': 50, #logging of the results
          'node_types': node_types,
          'edge_types': edge_types,
          'continuous_attributes': CONTINUOUS_ATTRIBUTES,
          'categorical_attributes': CATEGORICAL_ATTRIBUTES,
          'output_dir': f"./events/{time.time()}/"
          }           


tr_ge_graphs, tr_score, graphs_enc, input_graphs, target_graphs = go_train(train_graphs, tr_ge_split, save_fle = "test_model.ckpt", **kgcn_vars)

with session.transaction().write() as tx:
        write_predictions_to_grakn(tr_ge_graphs, tx, commit = False)  # Write predictions to grakn with learned probabilities
    
session.close()
client.close()

#val_ge_graphs, validation_evals = go_train(val_graphs, val_ge_split, reload_fle = "test_model.ckpt", **kgcn_vars)    
# Close transaction, session and client due to write query
    