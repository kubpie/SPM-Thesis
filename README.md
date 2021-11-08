# Design of a Graph Neural Network to predict the optimal resolution of the Sonar Performance Model
Master of Science Thesis by Jakub Pietrak

_Faculty of Mechanical, Maritime and Materials Engineering (3mE) · Delft University of
Technology_

## Foreword
This repository is an archive of code written for my graduation project carried out at TNO Acoustic & Sonar in Den Haag. 
Its main purpose is to present replicable results and to give an outline of proposed methods. Attached descriptions focus on the code functionality and hence they rarely dive into mathematical derivations or the decision process. For those interested, the complete report can be found [here](https://github.com/kubpie/SPM-Thesis/blob/master/docs/mscThesis.pdf "MSc Thesis Report").

## Project Introduction
**Graph Neural Networks** are a unique type of Deep Learning models that have capability to exploit an explicitly stated structure of data representation. By design they carry a strong _relational inductive bias_, which is a set of assumptions that makes the algorithm prioritize some solutions over another, independent of observed data. This makes the method especially interesting for applications to problems that are naturally relation-centric, or in which local interactions between features are the main value of interest. 
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/problem_setup.JPG" alt="drawing" width="700"/>
</p>

The presented research, aims to explore the potential of a GNN in application to an Ocean Acoustics problem, specifically to attempt a **prediction of the optimal sonar resolution**, calculated in a simulated ocean environment. 

**The database** consists of a aprox. 40,000 unique data points representing underwater sound propagation scenarios. Each scenario is described by a unique set of input parameters and a Sound Speed Profile (SSP) function. The inputs are limited to 4 numerical features - source depth, min. water depth, max. water dept and a slope gradient - describing problem geometry. There is one categorical feature that represents a bottom type, i.e. sand or mud. SSP has a critical impact on sound propagation mode, acting as a guideline for reflected and refracted paths of rays travelling through a water column. 

For each scenario the sound wave propagation is calculated with BELLHOP geometric ray-tracing algorithm. In the simulation run for any given scenario, the algorithm increases a number of emitted waves from a sonar aka source. It does so iteratively, until it finds the minimum number of waves that maps a water column with a desired accuracy, characterized by a stable Transmission Loss (TL). This number is saved as the optimal sonar resolution and the algorithm proceeds to the next scenario. <br />
This is the traditional way in which finding the sufficent number of waves requires multiple time-consuming numerical simulation runs. The performance of the algorithm is poor and thus its use is limited to offline applications. It cannot be used in i.e. real-time applications for onboard computing units of autnomous vehicles. This drawback creates a research gap for a data-driven model that could achieve the same result based simply on the numerical data, but without necessity to run iterative simulations.

A predictive model, which is to capture acoustic phenomena effectively, requires a mean of representing local interactions in a very scarce feature space which also cannot be trivially transformed into an input vector or a matrix for a machine learning model.

## General Approach & Folder Structure
The solution consists of the 3 main steps which correspond to folder structure of the presented repo:
1. [data_processing](https://github.com/kubpie/SPM-Thesis/tree/master/data_processing): data processig and exploratory data analysis 
2. [XGB](https://github.com/kubpie/SPM-Thesis/tree/master/XGB): development of a tradtional machine learning model in which performance indicates how well the problem could be solved using non-relational database structure
3. [KGCN](https://github.com/kubpie/SPM-Thesis/tree/master/KGCN): <br />
  a. development of a relational database in Grakn\Vaticle that includes expert knowledge of the phenomena of acoustic propagation, enriches data represenation and can potentially reinforce the learning <br />
  b. implementation of the Knowledge Graph Convolutional Network  that takes the graph representation as input and outputs a prediction for the optimal sonar resolution.
=======
### Recommended build:
Grakn Core 1.6.2
Python Client 1.6.1
Workbase 1.2.7

### Runtime instructions
To run GRAKN execute following steps:
1. download files and unpack to a single folder
2. start grakn server
3. upload the ssp_schema_kgcn.gql as --keyspcae ssp_schema
.\grakn console --keyspace ssp_schema --file path-to-folder\ssp_schema_kgcn.gql
4. In kgcn_data_migrate.py specify subset of the data you want to be migrated to grakn. Uploading the whole dataset will take a lot time.
5. use the query below in GRAKN Workbase (or flattened version directily in grakn console) to display the graph for each scenario $sid. 
The query takes a single input scenario_id $sid and should return a graph with all entities and attributes as in the draft block diagram. Replace $sid == {} with the scneario number.

### Troubleshooting
In case you run into issues with indices being out of range, I'd recommend updating your pandas to version 1.0.3 and\or numpy to 1.18.1.

### Query 
match        

$scn isa sound-propagation-scenario, has scenario_id $sid;
$ray isa ray-input, has num_rays $nray; 
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

$sid == {};

get; offset 0; limit 300;

#### Query flattened
match $scn isa sound-propagation-scenario, has scenario_id $sid; $ray isa ray-input, has num_rays $nray; $src isa source, has depth $dsrc; $seg isa bottom-segment, has depth $dseg, has length $l, has slope $s; $conv(converged_scenario: $scn, minimum_resolution: $ray) isa convergence;$srcp(defined_by_src: $scn, define_src: $src) isa src-position;$bathy(defined_by_bathy: $scn, define_bathy: $seg) isa bathymetry, has bottom_type $bt;$ssp isa SSP-vec, has location $loc, has season $ses, has SSP_value $sspval, has depth $dsspmax;$dct isa duct, has depth $ddct, has grad $gd; $speed(defined_by_SSP: $scn, define_SSP: $ssp) isa sound-speed;$duct(find_channel: $ssp, channel_exists: $dct) isa SSP-channel, has number_of_ducts $nod; $sspval has depth $dssp;{$dssp == $dsrc;} or {$dssp == $dseg;} or {$dssp == $ddct;} or {$dssp == $dsspmax;}; $sid == {}; get; offset 0; limit 300;
