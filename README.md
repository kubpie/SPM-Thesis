# SSP
To run GRAKN execute following steps:
1. download files and unpack to a single folder
2. start grakn server (you can used the commands in win_cmd_GRAKN.txt as a cheat sheet)
3. upload the ssp_schema.gql
4. in my_migrate.py change the path to the directory where the unpacked folder is stored 
5. run my_migrate from python interpreter or cmd. This will upload only a small fraction of selected data, shouldn't take longer than 2-3min.
6. use query from query.txt in GRAKN Workbase to display the graph for scenario. The query takes a single input scenario_id $sid and should return a graph with all entities and attributes as in the draft block diagram.
