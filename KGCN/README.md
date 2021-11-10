# Relational graph database design

[Grakn](https://en.wikipedia.org/wiki/GRAKN.AI) provides knowledge representation system with a transactional query interface. Hence it is required to process the queries and retrieve the graph data examples used in the learning process for the Graph Neural Network.

## Recommended build:
This Grakn-based GNN was developed to work with the following build: </br>
Grakn Core 1.6.2 </br>
Python Client 1.6.1 </br>
Grakn Workbase 1.2.7 </br>

## KGCN Schema
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/kgcn_schema.JPG" width = 900/>
</p>
The designed relational database schema to represent sound propagation. Scheam definiton can be found in ssp_schema_kgcn.gql.

## Runtime instructions

### Running GRAKN and investigating the knowledge graph:
1. download files and unpack to a single folder
2. start grakn server
3. upload the ssp_schema_kgcn.gql as --keyspcae ssp_schema
.\grakn console --keyspace ssp_schema --file path-to-folder\ssp_schema_kgcn.gql
4. In kgcn_data_migrate.py specify subset of the data you want to be migrated to grakn. Uploading the whole dataset will take a lot time.
5. use the query below in GRAKN Workbase (or flattened version directily in grakn console) to display the graph for each scenario $sid. 
The query takes a single input scenario_id $sid and should return a graph with all entities and attributes as in the draft block diagram. Replace $sid == {} with the scneario number.

#### Query 
match        

$scn isa sound-propagation-scenario, has scenario_id $sid;  </br>
$ray isa ray-input, has num_rays $nray;   </br>
$src isa source, has depth $dsrc;   </br>
$seg isa bottom-segment, has depth $dseg, has length $l, has slope $s;  </br>
$conv(converged_scenario: $scn, minimum_resolution: $ray) isa convergence;  </br>
$srcp(defined_by_src: $scn, define_src: $src) isa src-position;  </br>
$bathy(defined_by_bathy: $scn, define_bathy: $seg) isa bathymetry, has bottom_type $bt; </br>
$ssp isa SSP-vec, has location $loc, has season $ses, has SSP_value $sspval, has depth $dsspmax; </br>
$dct isa duct, has depth $ddct, has grad $gd;  </br>
$speed(defined_by_SSP: $scn, define_SSP: $ssp) isa sound-speed; </br>
$duct(find_channel: $ssp, channel_exists: $dct) isa SSP-channel, has number_of_ducts $nod;  </br>
 </br>
$sspval has depth $dssp; </br>
{$dssp == $dsrc;} or {$dssp == $dseg;} or {$dssp == $ddct;} or {$dssp == $dsspmax;}; </br>
$sid == {}; </br>
 </br>
get; offset 0; limit 300; 

# Knowldge Graph Convolutional Network
The KGCN model is designed to solve a relation prediction problem, which is a classification
task in a supervised learning setup. It should predict the existence of one of 16 possible
convergence relations associated with the correct resolution (number of rays) needed to solve
an example of a propagation scenario. This relation should be labelled as true, when its softmaxnormalized
probability value probability-exists is larger than probability-nonexists.
The opposite shall happen for the other negative candidate relations inserted by the conditional
rule presented. In the training loop the graphs are first split into training
and validation/generalisation sets. The GNN processes these simultaneously, calculating loss
and evaluation metrics for each set. Loss function compares the output graphs by comparing
probability values assigned to convergence nodes with one-hot encoded label that has a value
of 1 for the existing relation and 0 for the others. It does not take into account edges, or
pre-existing elements of the graph. The loss function used in the model is a softmax crossentropy
function, the same as the one used for the XGBoost model. Gradients of
the loss function are computed with Adam Optimizer algorithm and propagated back
to the neural network in the feedback loop. Every n-epoch number of executed training iterations
the feature vector outputted by the KGCN learner is used to calculate additional
evaluation metrics in training and generalisation sets. It used more strict condition on the
existence of each element by calculating the argmax() on the softmax probabilities output of
the learner. The argument with the maximum probability is assumed to be the existing one
and compared to the ground-truth graph. This sometimes causes some divergence between
the metrics and the loss function that calculates gradients based on the actual probability
values of each relation. The accuracy of labelling is averaged for the whole chunk of training
and validation data and output as: Ctr/Cge the percentage of correctly labelled nodes or
Str/Sge percentage of fully matching graphs:</br>
* Ctr/Cge - training/generalisation fraction of nodes/edges labelled correctly</br>
* Str/Sge - training/generalisation fraction of full examples solved correctly</br>
In that sense Str/Sge is more strict, as it requires the complete predicted graph to match
with the ground truth example, so all the nodes, both positive and negative ones, need to
be labelled correctly. Especially in the multi-class scenario it is easy to imagine that, while
the learner can usually easily exclude a few candidate relations, labelling of all 16 of these
correctly is a much harder task.
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/predicted_graph.jpg" width = 600/>
</p>

### KGCN Architecture
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/GN_block.jpg" width = 500/>
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/enc_dec.jpg" width = 300/>
</p>
Proposed KGCN used a fully-connected GN-block in the GN core block, defined by: </br>

1. update functions φ:
    * they update attribute values
    * are learned differentable functions
    * are independent models dedicated to separate graph elements

2. aggregation functions ρ:
    * reduce set of outputs from multiple graph elements of the same type
    * are invariant to permutations of their inputs
    * take variable number of inputs

GN core block learns how to pass useful updates around the graph. It passes the messages for _M_ steps before it updates the weights of graph elements. GN core uses concatenated input of _{t, t+1}_ steps which effectively makes it a **Messake Passing Neural Network**.</br>
The encoder block transforms Grakn output into NetworkX graph with defined transformation functions </br>
The decoder block contains a Neural Network which is a learned readout function. It learns how to represent aggregated update of each graph element after which it decodes it back to the initial Network graph format.

### Making predictions with KGCN

Training of a GNN is a time-consuming and computationally intensive process. To make predictions with KGCN execute the code in Google Colab notebook: [KGCN Colab Notebook](https://github.com/kubpie/SPM-Thesis/blob/master/JupyterNotebook/KGCNcolab.ipynb) and make use of free and awesome cloud GPU\TPU processing.

## Results
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/learning_example.jpg"/>
</p>
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/loss_func.jpg"/>
</p>
<p align="justify">
The next test was repeated in an unbiased 2-class setup. The input consisted of 1020 samples
of the 500 rays class and 1020 samples of the 1500 rays (the whole available sample for that
class). The model was trained for 103 iterations in the binary classification task with the best
parameters found in the biased dataset. The results show that the model can be progressively
learned towards achieving a approx. 0.73/0.7 Ctr/Cge accuracy score of the prediction and
above 0.5 Str/Sge accuracy of solving complete graphs while maintaining stability. What is
more, the loss and the evaluation metrics calculated in test and generalisation sets, converge
to very similar values throughout the the training progress which suggests that the model
does not have the tendency to overfit on the training set. Selected data at 0.8 tr/ge split
creates a representative training sample. Finally, it can be observed that the gradient of the
loss function gradually decreases by the end of the training, so hypothetically any further
improvement in the accuracy of predictions would require much longer training or increasing
the learning rate of the model, possibly even leading to overfitting at the training set. All of
the above observations support the conclusion that the model is indeed capable of learning
and solving a binary classification task on the BELLHOP database with certain, better than random
accuracy.
</p>
## Troubleshooting
In case you run into issues with indices being out of range, I'd recommend updating your pandas to version >1.0.3 and\or numpy to >1.18.1.


