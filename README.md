# SSP
To run GRAKN execute following steps:
1. download files and unpack to a single folder
2. start grakn server (you can used the commands in win_cmd_GRAKN.txt as a cheat sheet)
3. upload the ssp_schema.gql
4. in my_migrate.py change the path to the directory where the unpacked folder is stored 
5. run my_migrate from python interpreter or cmd. This will upload only a small fraction of selected data, shouldn't take longer than 2-3min.
6. use query from query.txt in GRAKN Workbase to display the graph for scenario. The query takes a single input scenario_id $sid and should return a graph with all entities and attributes as in the draft block diagram.


QUERY:
match
$scn isa sound-propagation-scenario, has scenario_id $sid;
$ray isa ray-input, has num_rays $nray; 
$src isa source, has depth $ds; 
$bs1 isa bottom-segment-1, has depth $dstart, has length $l;
$ssp isa SSP-vec, has location $loc, has season $ses, has mean_SSP $mssp, has stdev_SSP $sdssp, has mean_grad $mgrad, has stdev_grad $sdgrad, has SSP_value $sspval;
$conv($scn, $ray) isa convergence;
$srcp($scn, $src, $bs1) isa src-position;
$bathy($scn, $x) isa bathymetry, has bottom_type $bt;
$x has length $l, has attribute $a;
$speed($scn, $ssp) isa sound-speed;
$duct($ssp, $y) isa SSP-channel;
$y has attribute $b;
$sid == 1; #1, 440, 442, 9020

get 
$scn, $sid, $ray, $nray, $conv,
$src, $ds, $srcp, $bs1, $dstart, $l,
$bathy, $x, $bt, $a,
$ssp, $speed, $loc, $ses, $mssp, $sdssp, $mgrad, $sdgrad, $sspval,
$duct, $y, $b;
offset 0; limit 100;
