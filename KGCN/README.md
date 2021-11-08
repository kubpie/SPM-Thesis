# Knowldge Graph Convolutional Network


### Recommended build:
This Grakn-based GNN was developed to work with the following build: </br>
Grakn Core 1.6.2 </br>
Python Client 1.6.1 </br>
Grakn Workbase 1.2.7 </br>

[Grakn](https://en.wikipedia.org/wiki/GRAKN.AI) provides knowledge representation system with a transactional query interface. Hence it is required to process the queries and retrieve the graph data examples used in the learning process for the Graph Neural Network.

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

### Making predictions with KGCN
Training of a GNN is a time-consuming and computationally intensive process. To make predictions with KGCN execute the code in Google Colab notebook: [KGCN Colab Notebook](https://github.com/kubpie/SPM-Thesis/blob/master/KGCN/KGCNcolab.ipynb) and make use of GPU processing.

## Troubleshooting
In case you run into issues with indices being out of range, I'd recommend updating your pandas to version >1.0.3 and\or numpy to >1.18.1.


