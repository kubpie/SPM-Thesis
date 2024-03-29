#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import inspect
import time

from grakn.client import GraknClient

#from kglib.kgcn.pipeline.pipeline import pipeline
from pipeline_mod import pipeline

from kglib.utils.grakn.synthetic.examples.diagnosis.generate import generate_example_graphs
from kglib.utils.grakn.type.type import get_thing_types, get_role_types
from kglib.utils.graph.iterate import multidigraph_data_iterator
from kglib.utils.graph.query.query_graph import QueryGraph
from kglib.utils.graph.thing.queries_to_graph import build_graph_from_queries

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.reset_default_graph()

KEYSPACE = "diagnosis"
URI = "localhost:48555"

# Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to exist
PREEXISTS = 0

# Candidates are neither present in the input nor in the solution, they are negative samples
CANDIDATE = 1

# Elements to infer are the graph elements whose existence we want to predict to be true, they are positive samples
TO_INFER = 2

# Categorical Attribute types and the values of their categories
CATEGORICAL_ATTRIBUTES = {'name': ['Diabetes Type II', 'Multiple Sclerosis', 'Blurred vision', 'Fatigue', 'Cigarettes',
                                   'Alcohol']}
# Continuous Attribute types and their min and max values
CONTINUOUS_ATTRIBUTES = {'severity': (0, 1), 'age': (7, 80), 'units-per-week': (3, 29)}

TYPES_TO_IGNORE = ['candidate-diagnosis', 'example-id', 'probability-exists', 'probability-non-exists', 'probability-preexists']
ROLES_TO_IGNORE = ['candidate-patient', 'candidate-diagnosed-disease']

# The learner should see candidate relations the same as the ground truth relations, so adjust these candidates to
# look like their ground truth counterparts
TYPES_AND_ROLES_TO_OBFUSCATE = {'candidate-diagnosis': 'diagnosis',
                                'candidate-patient': 'patient',
                                'candidate-diagnosed-disease': 'diagnosed-disease'}


def diagnosis_example(num_graphs=200, #100
                      num_processing_steps_tr=5,
                      num_processing_steps_ge=5,
                      num_training_iterations=1000, #1000
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

    tr_ge_split = int(num_graphs*0.5)
    #TODO: #comment to run without creating new graphs
    #generate_example_graphs(num_graphs, keyspace=keyspace, uri=uri) #comment out this

    client = GraknClient(uri=uri)
    session = client.session(keyspace=keyspace)

    train_graphs = create_concept_graphs(list(range(num_graphs)), session)
    
        
    with session.transaction().read() as tx:
        # Change the terminology here onwards from thing -> node and role -> edge
        node_types = get_thing_types(tx)
        [node_types.remove(el) for el in TYPES_TO_IGNORE]

        edge_types = get_role_types(tx)
        [edge_types.remove(el) for el in ROLES_TO_IGNORE]
        print(f'Found node types: {node_types}')
        print(f'Found edge types: {edge_types}')
        
    kgcn_vars = {
    'num_processing_steps_tr': 5,
    'num_processing_steps_ge': 5,
    'num_training_iterations': 1000,
    'learning_rate': 1e-2, #added to tube
    'latent_size': 16, #MLP param
    'num_layers': 2, #MLP param
    'weighted': False, #loss function modification
    'log_every_epochs': 50, #logging of the results
    'clip': 5, #gradient clipping 5
    'node_types': node_types,
    'edge_types': edge_types,
    'continuous_attributes': CONTINUOUS_ATTRIBUTES,
    'categorical_attributes': CATEGORICAL_ATTRIBUTES,
    'output_dir': f"./events/diagnosis/{time.time()}/",
    'save_fle': "diag_summ.ckpt"
    }   
    """
    ge_graphs, solveds_tr, solveds_ge = pipeline(graphs,
                                                 tr_ge_split,
                                                 node_types,
                                                 edge_types,
                                                 num_processing_steps_tr=num_processing_steps_tr,
                                                 num_processing_steps_ge=num_processing_steps_ge,
                                                 num_training_iterations=num_training_iterations,
                                                 continuous_attributes=CONTINUOUS_ATTRIBUTES,
                                                 categorical_attributes=CATEGORICAL_ATTRIBUTES,
                                                 output_dir=f"./events/{time.time()}/",
                                                 do_test = False,
                                                 save_fle = save_fle,
                                                 reload_fle = "")
    """

    ge_graphs, solveds_tr, solveds_ge = pipeline(graphs = train_graphs,             
                                                tr_ge_split = tr_ge_split,                         
                                                do_test = False,
                                                **kgcn_vars)
    
    #with session.transaction().write() as tx:
    #    write_predictions_to_grakn(ge_graphs, tx)

    session.close()
    client.close()

    return ge_graphs, solveds_tr, solveds_ge


def create_concept_graphs(example_indices, grakn_session):
    """
    Builds an in-memory graph for each example, with an example_id as an anchor for each example subgraph.
    Args:
        example_indices: The values used to anchor the subgraph queries within the entire knowledge graph
        grakn_session: Grakn Session

    Returns:
        In-memory graphs of Grakn subgraphs
    """

    graphs = []
    infer = True

    for example_id in example_indices:
        print(f'Creating graph for example {example_id}')
        graph_query_handles = get_query_handles(example_id)
        with grakn_session.transaction().read() as tx:
            # Build a graph from the queries, samplers, and query graphs
            graph = build_graph_from_queries(graph_query_handles, tx, infer=infer)

        obfuscate_labels(graph, TYPES_AND_ROLES_TO_OBFUSCATE)

        graph.name = example_id
        graphs.append(graph)

    return graphs


def obfuscate_labels(graph, types_and_roles_to_obfuscate):
    # Remove label leakage - change type labels that indicate candidates into non-candidates
    for data in multidigraph_data_iterator(graph):
        for label_to_obfuscate, with_label in types_and_roles_to_obfuscate.items():
            if data['type'] == label_to_obfuscate:
                data.update(type=with_label)
                break


def get_query_handles(example_id):
    """
    Creates an iterable, each element containing a Graql query, a function to sample the answers, and a QueryGraph
    object which must be the Grakn graph representation of the query. This tuple is termed a "query_handle"

    Args:
        example_id: A uniquely identifiable attribute value used to anchor the results of the queries to a specific
                    subgraph

    Returns:
        query handles
    """

    # === Hereditary Feature ===
    hereditary_query = inspect.cleandoc(f'''match
           $p isa person, has example-id {example_id};
           $par isa person;
           $ps(child: $p, parent: $par) isa parentship;
           $diag(patient:$par, diagnosed-disease: $d) isa diagnosis;
           $d isa disease, has name $n;
           get;''')

    vars = p, par, ps, d, diag, n = 'p', 'par', 'ps', 'd', 'diag', 'n'
    hereditary_query_graph = (QueryGraph()
                              .add_vars(vars, PREEXISTS)
                              .add_role_edge(ps, p, 'child', PREEXISTS)
                              .add_role_edge(ps, par, 'parent', PREEXISTS)
                              .add_role_edge(diag, par, 'patient', PREEXISTS)
                              .add_role_edge(diag, d, 'diagnosed-disease', PREEXISTS)
                              .add_has_edge(d, n, PREEXISTS))

    # === Consumption Feature ===
    consumption_query = inspect.cleandoc(f'''match
           $p isa person, has example-id {example_id};
           $s isa substance, has name $n;
           $c(consumer: $p, consumed-substance: $s) isa consumption, 
           has units-per-week $u; get;''')

    vars = p, s, n, c, u = 'p', 's', 'n', 'c', 'u'
    consumption_query_graph = (QueryGraph()
                               .add_vars(vars, PREEXISTS)
                               .add_has_edge(s, n, PREEXISTS)
                               .add_role_edge(c, p, 'consumer', PREEXISTS)
                               .add_role_edge(c, s, 'consumed-substance', PREEXISTS)
                               .add_has_edge(c, u, PREEXISTS))

    # === Age Feature ===
    person_age_query = inspect.cleandoc(f'''match 
            $p isa person, has example-id {example_id}, has age $a; 
            get;''')

    vars = p, a = 'p', 'a'
    person_age_query_graph = (QueryGraph()
                              .add_vars(vars, PREEXISTS)
                              .add_has_edge(p, a, PREEXISTS))

    # === Risk Factors Feature ===
    risk_factor_query = inspect.cleandoc(f'''match 
            $d isa disease; 
            $p isa person, has example-id {example_id}; 
            $r(person-at-risk: $p, risked-disease: $d) isa risk-factor; 
            get;''')

    vars = p, d, r = 'p', 'd', 'r'
    risk_factor_query_graph = (QueryGraph()
                               .add_vars(vars, PREEXISTS)
                               .add_role_edge(r, p, 'person-at-risk', PREEXISTS)
                               .add_role_edge(r, d, 'risked-disease', PREEXISTS))

    # === Symptom ===
    vars = p, s, sn, d, dn, sp, sev, c = 'p', 's', 'sn', 'd', 'dn', 'sp', 'sev', 'c'

    symptom_query = inspect.cleandoc(f'''match
           $p isa person, has example-id {example_id};
           $s isa symptom, has name $sn;
           $d isa disease, has name $dn;
           $sp(presented-symptom: $s, symptomatic-patient: $p) isa symptom-presentation, has severity $sev;
           $c(cause: $d, effect: $s) isa causality;
           get;''')

    symptom_query_graph = (QueryGraph()
                           .add_vars(vars, PREEXISTS)
                           .add_has_edge(s, sn, PREEXISTS)
                           .add_has_edge(d, dn, PREEXISTS)
                           .add_role_edge(sp, s, 'presented-symptom', PREEXISTS)
                           .add_has_edge(sp, sev, PREEXISTS)
                           .add_role_edge(sp, p, 'symptomatic-patient', PREEXISTS)
                           .add_role_edge(c, s, 'effect', PREEXISTS)
                           .add_role_edge(c, d, 'cause', PREEXISTS))

    # === Diagnosis ===

    diag, d, p, dn = 'diag', 'd', 'p', 'dn'

    diagnosis_query = inspect.cleandoc(f'''match
           $p isa person, has example-id {example_id};
           $d isa disease, has name $dn;
           $diag(patient: $p, diagnosed-disease: $d) isa diagnosis;
           get;''')

    diagnosis_query_graph = (QueryGraph()
                             .add_vars([diag], TO_INFER)
                             .add_vars([d, p, dn], PREEXISTS)
                             .add_role_edge(diag, d, 'diagnosed-disease', TO_INFER) #TO_INFER VS CANDIDATE BELOW
                             .add_role_edge(diag, p, 'patient', TO_INFER))

    # === Candidate Diagnosis ===
    candidate_diagnosis_query = inspect.cleandoc(f'''match
           $p isa person, has example-id {example_id};
           $d isa disease, has name $dn;
           $diag(candidate-patient: $p, candidate-diagnosed-disease: $d) isa candidate-diagnosis; 
           get;''')

    candidate_diagnosis_query_graph = (QueryGraph()
                                       .add_vars([diag], CANDIDATE)
                                       .add_vars([d, p, dn], PREEXISTS)
                                       .add_role_edge(diag, d, 'candidate-diagnosed-disease', CANDIDATE)
                                       .add_role_edge(diag, p, 'candidate-patient', CANDIDATE))

    return [
        (symptom_query, lambda x: x, symptom_query_graph),
        (diagnosis_query, lambda x: x, diagnosis_query_graph),
        (candidate_diagnosis_query, lambda x: x, candidate_diagnosis_query_graph),
        (risk_factor_query, lambda x: x, risk_factor_query_graph),
        (person_age_query, lambda x: x, person_age_query_graph),
        (consumption_query, lambda x: x, consumption_query_graph),
        (hereditary_query, lambda x: x, hereditary_query_graph)
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
                if concept_type == 'diagnosis' or concept_type == 'candidate-diagnosis':
                    neighbours = graph.neighbors(node)

                    for neighbour in neighbours:
                        concept = graph.nodes[neighbour]['concept']
                        if concept.type_label == 'person':
                            person = concept
                        else:
                            disease = concept

                    p = data['probabilities']
                    query = (f'match'
                             f'$p id {person.id};'
                             f'$d id {disease.id};'
                             f'$kgcn isa kgcn;'
                             f'insert'
                             f'$pd(patient: $p, diagnosed-disease: $d, diagnoser: $kgcn) isa diagnosis,'
                             f'has probability-exists {p[2]:.3f},'
                             f'has probability-non-exists {p[1]:.3f},'  
                             f'has probability-preexists {p[0]:.3f};')
                    tx.query(query)
    tx.commit()


if __name__ == "__main__":
    ge_graphs, solveds_tr, solveds_ge, graphs_enc, input_graphs, target_graphs = diagnosis_example()
