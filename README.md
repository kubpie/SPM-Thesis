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
match $scn isa sound-propagation-scenario, has scenario_id 440; $ray isa ray-input, has num_rays $nray; $src isa source, has depth $dsrc; $seg isa bottom-segment, has depth $dseg, has length $l, has slope $s; $conv(converged_scenario: $scn, minimum_resolution: $ray) isa convergence;$srcp(defined_by_src: $scn, define_src: $src) isa src-position;$bathy(defined_by_bathy: $scn, define_bathy: $seg) isa bathymetry, has bottom_type $bt;$ssp isa SSP-vec, has location $loc, has season $ses, has SSP_value $sspval, has depth $dsspmax;$dct isa duct, has depth $ddct, has grad $gd; $speed(defined_by_SSP: $scn, define_SSP: $ssp) isa sound-speed;$duct(find_channel: $ssp, channel_exists: $dct) isa SSP-channel, has number_of_ducts $nod; $sspval has depth $dssp;{$dssp == $dsrc;} or {$dssp == $dseg;} or {$dssp == $ddct;} or {$dssp == $dsspmax;};get; offset 0; limit 300;
