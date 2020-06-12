### Recommended build:
Grakn Core 1.6.2
Python Client 1.6.1
Workbase 1.2.7

### Runtime instructions
To run GRAKN execute following steps:
1. download files and unpack to a single folder
2. start grakn server
3. upload the ssp_schema.gql
4. run my_migrate from python interpreter or cmd. This will upload only a small fraction of selected data, shouldn't take longer than 2-3min.
5. use the query below in GRAKN Workbase (or flattened version directily in grakn console) to display the graph for scenario.
The query takes a single input scenario_id $sid and should return a graph with all entities and attributes as in the draft block diagram.

### Query 
match
$scn isa sound-propagation-scenario, has scenario_id $sid;
$ray isa ray-input, has num_rays $nray; 
$src isa source, has depth $ds; 
$bs1 isa bottom-segment-1, has depth $dstart, has length $l;
$ssp isa SSP-vec, has location $loc, has season $ses, has SSP_value $sspval, has depth $dsspmax;

$conv($scn, $ray) isa convergence;
$srcp($scn, $src, $bs1) isa src-position;
$bathy($scn, $x) isa bathymetry, has bottom_type $bt;
$x has attribute $a;
$speed($scn, $ssp) isa sound-speed;
$duct($ssp, $y) isa SSP-channel;
$y has attribute $b;

$sspval has depth $dssp;
{$dssp == $ds;} or {$dssp == $dstart;} or {$dssp == $a;} or {$dssp == $b;} or {$dssp == $dsspmax;} ;

$sid == 440; #1,440,442,9020

get 
$scn, $sid, $ray, $nray, $conv,
$src, $ds, $srcp, $bs1, $dstart, $l,
$bathy, $x, $bt, $a,
$ssp, $speed, $loc, $ses, $sspval, $dsspmax, $dssp,
$duct, $y, $b;

offset 0; limit 150;

#### Query flattened:
match $scn isa sound-propagation-scenario, has scenario_id $sid; $ray isa ray-input, has num_rays $nray; $src isa source, has depth $ds; $bs1 isa bottom-segment-1, has depth $dstart, has length $l; $ssp isa SSP-vec, has location $loc, has season $ses, has SSP_value $sspval, has depth $dsspmax; $conv($scn, $ray) isa convergence; $srcp($scn, $src, $bs1) isa src-position; $bathy($scn, $x) isa bathymetry, has bottom_type $bt; $x has attribute $a; $speed($scn, $ssp) isa sound-speed; $duct($ssp, $y) isa SSP-channel; $y has attribute $b; $sspval has depth $dssp; {$dssp == $ds;} or {$dssp == $dstart;} or {$dssp == $a;} or {$dssp == $b;} or {$dssp == $dsspmax;}; $sid == 440; get $scn, $sid, $ray, $nray, $conv,$src, $ds, $srcp, $bs1, $dstart, $l, $bathy, $x, $bt, $a, $ssp, $speed, $loc, $ses, $sspval, $dsspmax, $dssp,$duct, $y, $b; offset 0; limit 150;