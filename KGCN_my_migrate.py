# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:20:55 2020

@author: kubap
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:32:19 2020

@author: kubap
"""
from grakn.client import GraknClient
import numpy as np
import pandas as pd
#import sys
#sys.path.append(r'C:\Users\kubap\Documents\THESIS\DATA_PREP')
from ssp_features import SSPGrad, SSPStat, SSPId
from data_prep import LoadData, FeatDuct, FeatBathy, FeatSSPId, FeatSSPOnDepth
from decimal import Decimal

# 1. Insert ENTITIES  
# 2. Insert RELATIONS
# 3. In a long run, take a look at tube_network example for concurrent insert

# Query tips:
# * string-vars need to be put in "apostrophe", numericals don't
# * mind spaces and semi-colons
# * input files need to be saved as .csv comma-delimited (not UTF-8 as it gives BOM at the start of the file )
# * You can create query for ONLY ONE entity/relation at a time to avoid Unique-ID conflicts
#   while committing the transaction later


def build_graph(Inputs, keyspace_name):
    """
      gets the job done:
      1. creates a Grakn instance
      2. creates a session to the targeted keyspace
      3. for each input:
        - a. constructs the full path to the data file
        - b. loads csv to Grakn
      :param input as list of dictionaties: each dictionary contains details required to parse the data
    """
    with GraknClient(uri="localhost:48555") as client:
        with client.session(keyspace=keyspace_name) as session:
            for Input in Inputs:
                load_data_into_grakn(Input, session)

def load_data_into_grakn(Input, session):
    for Node in Input:
        it = 0
        Node['QueryListUnique'] = remove_duplicates(Node)
        print(f"#### Loading [{Node['NodeName']}] queries ####\n\n")
        for query in Node['QueryListUnique']:
            it += 1
            #one query per transaction
            with session.transaction().write() as transaction:
                if query: #if query is not empty
                    print("Executing Graql Query #" + str(it) + ": " + query + "\n")
                    transaction.query(query)  
                    transaction.commit() 
        print(f"\n #### Inserted {it} instances for the node [{Node['NodeName']}] into Grakn.#### \n")

def remove_duplicates(Node):
    unique_list = []
    for elem in Node['QueryList']:
        if elem not in unique_list:
            unique_list.append(elem)
    duplicates_nr = len(Node['QueryList']) - len(unique_list)
    print(f'Removed {duplicates_nr} duplicates from [{Node["NodeName"]}]')
    return unique_list
########################################################
### Data loading functions library for ENTITIY nodes ###
########################################################

def Scenario_inner(idx, query_type = 'insert'):
    graql_insert_query = query_type
    graql_insert_query += ' $scn isa sound-propagation-scenario'
    graql_insert_query += f', has scenario_id {idx};'
    return graql_insert_query

def Scenario(data):
    graql_queries = []
    for idx in data.index:    
        graql_queries.append(Scenario_inner(idx, query_type = 'insert'))
    return graql_queries
   
def RayInput_inner(ray, query_type = 'insert'):
    graql_insert_query = query_type
    graql_insert_query += ' $ray isa ray-input'
    graql_insert_query += f', has num_rays {ray};'
    return graql_insert_query

def RayInput(data):
    graql_queries = []
    ray_input = np.unique(data['num_rays'])
    for ray in ray_input:      
        graql_queries.append(RayInput_inner(ray, query_type = 'insert'))
    return graql_queries    
    
#bottom seg. 1 appears in each scenario
def BottomSegment_ALL(Bathy):
# Using method of list comprehensions (faster then iterrows) get queries     
    graql_queries = []
    for dstart, lenflat in zip(Bathy['d_start'], Bathy['len_flat']):
        graql_queries.append(BottomSegment1_inner(dstart, lenflat))
    for dend, lenflat, slope in zip(Bathy['d_end'], Bathy['len_flat'],  Bathy['slope']):
        if slope != 0:
            graql_queries.append(BottomSegment2_inner(dend, lenflat))
    for dstart, dend, lenslope, slope in zip(Bathy['d_start'],Bathy['d_end'], Bathy['len_slope'], Bathy['slope']):
        if slope != 0:          
            graql_queries.append(WedgeSegment_inner(dstart, dend, lenslope, slope))
    return graql_queries

def BottomSegment1_inner(dstart, lenflat, query_type = 'insert'):
    graql_insert_query = query_type
    graql_insert_query += ' $bs isa bottom-segment'
    graql_insert_query += ', has depth ' + f'{dstart}'
    graql_insert_query += ', has length ' +  f'{lenflat}'
    graql_insert_query += ', has slope ' +  str(0)
    graql_insert_query += ';'
    return graql_insert_query       
    
#bottom seg. 1 appears in each scenario
def BottomSegment1(Bathy):
# Using method of list comprehensions (faster then iterrows) get queries     
    graql_queries = []
    for dstart, lenflat in zip(Bathy['d_start'], Bathy['len_flat']):
        graql_queries.append(BottomSegment1_inner(dstart, lenflat))
    return graql_queries

def BottomSegment2_inner(dend, lenflat, query_type = 'insert'):
    graql_insert_query = query_type
    graql_insert_query += ' $bs isa bottom-segment'
    graql_insert_query += ', has depth '  + f'{dend}'
    graql_insert_query += ', has length ' + f'{lenflat}'
    graql_insert_query += ', has slope ' +  str(0)
    graql_insert_query += ';'  
    return graql_insert_query

def BottomSegment2(Bathy):
    #bottom seg. 2 only for sloped scn
    graql_queries = []
    for dend, lenflat, slope in zip(Bathy['d_end'], Bathy['len_flat'],  Bathy['slope']):
        if slope != 0:
            graql_queries.append(BottomSegment2_inner(dend, lenflat))
    return graql_queries

def WedgeSegment_inner(dstart, dend, lenslope, slope, query_type = 'insert'):
    graql_insert_query = query_type
    graql_insert_query += ' $ws isa bottom-segment'
    graql_insert_query += ', has depth ' + f'{(dstart+dend)/2}'
    #graql_insert_query += ', has depth '  + f'{dend}'
    graql_insert_query += ', has slope ' + f'{slope}'
    graql_insert_query += ', has length ' + f'{lenslope}'
    graql_insert_query += ';'
    return graql_insert_query

def WedgeSegment(Bathy):
    #wedge segment only for sloped scn
    #it needs to check only one type, as the other is symmetric
    graql_queries = []
    for dstart, dend, lenslope, slope in zip(Bathy['d_start'],Bathy['d_end'], Bathy['len_slope'], Bathy['slope']):
        if slope != 0:          
            graql_queries.append(WedgeSegment_inner(dstart, dend, lenslope, slope))
    return graql_queries


def SSPVec_inner(ssp, dmax, SSP_Input, SSP_Stat, query_type = 'insert'):
    #TODO: Load from pre-formatted data file, such that SSP-val are already pre-selected for each scenario
    #       The filter at the end should remove all doubles happpening in the process
    
    #set floating-point numbers precision
    precision = 10    
    
    depth = SSP_Input['DEPTH'].values.tolist()
    maxdepth = depth
    
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    season = next((s for s in seasons if s in ssp), False)
    location = ssp.replace(season, '')[:-1]
    location = location.replace(' ', '-')
    
    graql_insert_query = query_type
    graql_insert_query += ' $ssp isa SSP-vec'
    graql_insert_query += ', has location "' + location + '"'
    graql_insert_query += ', has season "' + season + '"'
    graql_insert_query += f', has depth {dmax}' 
    
    
    # TODO: Change stat to True if you want to add statistical data on SSP
    stat = False
    
    #location, season and max_depth should be enough to match on a single unique SSP 
    if query_type == 'insert':
        if stat == True:
            for stat in SSP_Stat[ssp].columns:
                sval = SSP_Stat[ssp][stat].iloc[depth.index(dmax)]
                graql_insert_query += f', has {stat} {Decimal(sval):.{precision}}'
        # Where it's necessary the new Python 3.6 f-formatting replaces orginal str input to set the {precision} and avoid
        # Scientific notation input (such as 5.6e-04).
        for md in maxdepth[:maxdepth.index(dmax)+1]:
            graql_insert_query += ', has SSP_value ' + f'{SSP_Input[ssp].iloc[depth.index(md)]:.{precision}}'
        graql_insert_query += ';'
    else:
        graql_insert_query += ';'
        
    return graql_insert_query

def SSPVec(SSP_Input, SSP_Stat):  
    graql_queries = []
    maxdepth = SSP_Input["DEPTH"]
    for ssp in SSP_Input.iloc[:,1:].columns:
        for dmax in maxdepth:
            graql_queries.append(SSPVec_inner(ssp, dmax, SSP_Input, SSP_Stat))    
    return graql_queries


def Source(data):
    graql_queries = []
    source_depth = np.unique(data['source_depth'])  
    for src in source_depth:
        graql_queries.append(Source_inner(src, query_type = 'insert'))
    return graql_queries

def Source_inner(src, query_type = 'insert'):
    graql_insert_query = query_type
    graql_insert_query += ' $src isa source'
    graql_insert_query += f', has depth {src};'
    return graql_insert_query

def Sonic_Layer_inner(sld, sldgrad, query_type = 'insert'):    
    precision = 10
    graql_insert_query = query_type
    graql_insert_query += ' $sld isa duct'
    graql_insert_query += f', has depth {sld}'
    graql_insert_query += f', has grad {Decimal(sldgrad):.{precision}}'
    graql_insert_query += f', has duct_type "SLD"'
    graql_insert_query += ';'
    return graql_insert_query

def SonicLayer(SSP_Prop):
    graql_queries = []
    for sld, sldgrad in zip(SSP_Prop['SLD_depth'],SSP_Prop['SLD_avgrad']):
        if np.isnan(sld) == False:
            graql_queries.append(Sonic_Layer_inner(sld, sldgrad, query_type = 'insert'))
    return graql_queries

def DeepChannel_inner(dcax, dctop, dcbot, dcgtop, dcgbot, query_type = 'insert'):
    precision = 10
    graql_insert_query = query_type
    graql_insert_query += ' $dc isa duct'
    graql_insert_query += f', has depth {dcax}'
    graql_insert_query += f', has depth {dctop}'
    graql_insert_query += f', has depth {dcbot}'
    graql_insert_query += f', has grad {Decimal(dcgtop):.{precision}}'
    graql_insert_query += f', has grad {Decimal(dcgbot):.{precision}}'
    graql_insert_query += f', has duct_type "DC"'
    graql_insert_query += ';'
    return graql_insert_query

def DeepChannel(SSP_Prop):
    graql_queries = []
    for dcax, dctop, dcbot, dcgtop, dcgbot in zip(SSP_Prop['DC_axis'], \
    SSP_Prop['DC_top'], SSP_Prop['DC_bot'], SSP_Prop['DC_avgrad_top'], SSP_Prop['DC_avgrad_bot']):
        if np.isnan(dcax) == False:
            graql_queries.append(DeepChannel_inner(dcax, dctop, dcbot, dcgtop, dcgbot, query_type = 'insert'))
    return graql_queries
        
def DuctExists():
    graql_queries = []
    number_of_ducts = [0]
    for nod in number_of_ducts:
        graql_queries.append(DuctExists_inner(query_type = 'insert'))
    return graql_queries

def DuctExists_inner(query_type = 'insert'):
    graql_insert_query = query_type
    graql_insert_query += f' $dex isa duct, has depth 0, has grad 0, has duct_type "None";'
    return graql_insert_query
    
#########################################################
### Data loading functions library for RELATION nodes ###
#########################################################

def rel_Convergence(data):
    graql_queries = []
    for idx, ray in zip(data.index, data['num_rays']):
        graql_insert_query = Scenario_inner(idx, query_type = 'match')
        graql_insert_query += RayInput_inner(ray, query_type = '')
        graql_insert_query += (' insert $conv(converged_scenario: $scn, minimum_resolution: $ray) isa convergence;')
        graql_queries.append(graql_insert_query)
    return graql_queries

def rel_SrcPosition(data, Bathy):
    graql_queries = []
    for idx, src, slope, dmin, dmax in \
    zip(data.index, data['source_depth'], data['wedge_slope'], \
        data['water_depth_min'], data['water_depth_max']):
        graql_insert_query = Scenario_inner(idx, query_type = 'match')
        graql_insert_query += Source_inner(src, query_type = '')
        #if slope == 0 or slope == -2:
        #    dstart = dmin
        #    dend = dmax
        #else:
        #    dstart = dmax
        #    dend = dmin    
        #find_lenflat = Bathy.loc[(Bathy['d_start'] == dstart) & (Bathy['d_end'] == dend), 'len_flat']
        #lenflat = find_lenflat.values[0]        
        #graql_insert_query += BottomSegment1_inner(dstart, lenflat, query_type = '')
        graql_insert_query += (' insert $srcp(define_src: $src, defined_by_src: $scn) isa src-position;')
        graql_queries.append(graql_insert_query)
    return graql_queries
   

def rel_Bathymetry(data, Bathy):
    graql_queries = []
    for idx, src, btype, slope, dmin, dmax in \
    zip(data.index, data['source_depth'], data['bottom_type'], data['wedge_slope'], \
        data['water_depth_min'], data['water_depth_max']):
        
        if dmin == dmax:
            lenflat = 44000
            graql_insert_query = Scenario_inner(idx, query_type = 'match')
            graql_insert_query += BottomSegment1_inner(dmin, lenflat, query_type = '')
            graql_insert_query += ' insert $bathy(define_bathy: $bs, defined_by_bathy: $scn) isa bathymetry;'       
            graql_insert_query += f' $bathy has bottom_type {btype};'
            graql_queries.append(graql_insert_query)
        else:
            if slope == 2:
                dstart = dmax
                dend = dmin 
                       
            if slope == -2:
                dstart = dmin
                dend = dmax
         
            find_lenflat = Bathy.loc[(Bathy['d_start'] == dstart) & (Bathy['d_end'] == dend), 'len_flat']
            lenflat = find_lenflat.values[0]        
            find_lenslope = Bathy.loc[(Bathy['d_start'] == dstart) & (Bathy['d_end'] == dend) & (Bathy['slope'] == slope), 'len_slope']
            lenslope = find_lenslope.values[0]
            
            graql_insert_query = Scenario_inner(idx, query_type = 'match')
            graql_insert_query += BottomSegment1_inner(dstart, lenflat, query_type = '')
            graql_insert_query += ' insert $bathy(define_bathy: $bs, defined_by_bathy: $scn) isa bathymetry;'       
            graql_insert_query += f' $bathy has bottom_type {btype};'
            graql_queries.append(graql_insert_query)
            
            graql_insert_query = Scenario_inner(idx, query_type = 'match')
            graql_insert_query += BottomSegment2_inner(dend, lenflat, query_type = '')
            graql_insert_query += ' insert $bathy(define_bathy: $bs, defined_by_bathy: $scn) isa bathymetry;'       
            graql_insert_query += f' $bathy has bottom_type {btype};'
            graql_queries.append(graql_insert_query)
            
            graql_insert_query = Scenario_inner(idx, query_type = 'match')
            graql_insert_query += WedgeSegment_inner(dstart, dend, lenslope, slope, query_type = '')
            graql_insert_query += ' insert $bathy(define_bathy: $ws, defined_by_bathy: $scn) isa bathymetry;'       
            graql_insert_query += f' $bathy has bottom_type {btype};'
            graql_queries.append(graql_insert_query)
    return graql_queries

def rel_SoundSpeed(data, SSP_Input, SSP_Stat):
    graql_queries = []
    for idx, ssp, dmax in zip(data.index, data['profile'], data['water_depth_max']): 
        graql_insert_query = Scenario_inner(idx, query_type = 'match')
        graql_insert_query += SSPVec_inner(ssp, dmax, SSP_Input, SSP_Stat, query_type = '')
        graql_insert_query += ' insert $bathy(define_SSP: $ssp, defined_by_SSP: $scn) isa sound-speed;'
        graql_queries.append(graql_insert_query)
    return graql_queries


def rel_SSPChannel(SSP_Input, SSP_Stat, SSP_Prop):
    graql_queries = []
    for index, row in SSP_Prop.iterrows():
        graql_insert_query = SSPVec_inner(row['SSP'], row['dmax'], SSP_Input, SSP_Stat, query_type = 'match')
        #if (row['SLD_depth'] == None and row['DC_axis'] == None):
        
        if (np.isnan(row['SLD_depth']) and np.isnan(row['DC_axis'])):
            nod = 0
            graql_insert_query = SSPVec_inner(row['SSP'], row['dmax'], SSP_Input, SSP_Stat, query_type = 'match')
            graql_insert_query += DuctExists_inner(query_type = '')
            graql_insert_query += f' insert $sspch(find_channel: $ssp, channel_exists: $dex) isa SSP-channel, has number_of_ducts {nod};'
            graql_queries.append(graql_insert_query)
        else:
            if np.isnan(row['SLD_depth']) == False and np.isnan(row['DC_axis']) == True:
                nod = 1
                graql_insert_query = SSPVec_inner(row['SSP'], row['dmax'], SSP_Input, SSP_Stat, query_type = 'match')
                graql_insert_query += Sonic_Layer_inner(row['SLD_depth'], row['SLD_avgrad'], query_type = '')
                graql_insert_query += f' insert $sspch(find_channel: $ssp, channel_exists: $sld) isa SSP-channel, has number_of_ducts {nod};'
                graql_queries.append(graql_insert_query)
                
            if np.isnan(row['DC_axis']) == False and np.isnan(row['SLD_depth']) == True:
                nod = 1
                graql_insert_query = SSPVec_inner(row['SSP'], row['dmax'], SSP_Input, SSP_Stat, query_type = 'match')
                graql_insert_query += DeepChannel_inner(row['DC_axis'], row['DC_top'], row['DC_bot'], row['DC_avgrad_top'], row['DC_avgrad_bot'], query_type = '')
                graql_insert_query += f' insert $sspch(find_channel: $ssp, channel_exists: $dc) isa SSP-channel, has number_of_ducts {nod};'
                graql_queries.append(graql_insert_query)
            if np.isnan(row['SLD_depth']) == False and np.isnan(row['DC_axis']) == False:
                nod = 2
                graql_insert_query = SSPVec_inner(row['SSP'], row['dmax'], SSP_Input, SSP_Stat, query_type = 'match')
                graql_insert_query += Sonic_Layer_inner(row['SLD_depth'], row['SLD_avgrad'], query_type = '')
                graql_insert_query += f' insert $sspch(find_channel: $ssp, channel_exists: $sld) isa SSP-channel, has number_of_ducts {nod};'
                graql_queries.append(graql_insert_query)
                graql_insert_query = SSPVec_inner(row['SSP'], row['dmax'], SSP_Input, SSP_Stat, query_type = 'match')
                graql_insert_query += DeepChannel_inner(row['DC_axis'], row['DC_top'], row['DC_bot'], row['DC_avgrad_top'], row['DC_avgrad_bot'], query_type = '')
                graql_insert_query += f' insert $sspch(find_channel: $ssp, channel_exists: $dc) isa SSP-channel, has number_of_ducts {nod};'
                graql_queries.append(graql_insert_query)

    return graql_queries  

def rel_SSPvecToDepth(SSP_Input):
    graql_queries = []
    precision = 10
    for column in SSP_Input.columns.tolist()[1:]:
        for depth, sspval in zip(SSP_Input['DEPTH'],SSP_Input[column]):
            graql_insert_query = f'match $sspval isa SSP_value;'
            graql_insert_query += f' $sspval == {sspval:.{precision}};'
            graql_insert_query += f' insert $sspval has depth {depth};'
            graql_queries.append(graql_insert_query)

    return graql_queries
            
###########################################
###     Build input data dictionaries   ###
###########################################
import os
path = os.getcwd()+'\data\\'

Bathy = pd.read_excel(path+"env.xlsx", sheet_name = "BATHY")
SSP_Input = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
#SSP_Grad = SSPGrad(SSP_Input, path, save = False
SSP_Stat = pd.read_excel(path+"env.xlsx", sheet_name = "SSP_STAT")#SSPStat(SSP_Input, path, plot = False, save = False)
SSP_Prop = pd.read_excel(path+"env.xlsx", sheet_name = "SSP_PROP")#SSPId(SSP_Input, path, plot = False, save = False)

raw_data = LoadData(path)
data = FeatDuct(raw_data, Input_Only = True) #leave only model input
data_complete = pd.read_csv(path+"data_complete.csv")

# DATA SELECTION FOR GRAKN TESTING
data = pd.concat([data.iloc[0:10,:],data.iloc[440:446,:],data.iloc[9020:9026,:]])
ssp_select = ["Mediterranean Sea Winter","Mediterranean Sea Spring","South Pacific Ocean Spring"]
SSP_Stat[ssp_select][:]
SSP_Prop = SSP_Prop[(SSP_Prop['SSP'] == "Mediterranean Sea Winter") | (SSP_Prop['SSP'] == "Mediterranean Sea Spring") | (SSP_Prop['SSP'] == "South Pacific Ocean Spring")]
SSP_Input = SSP_Input.loc[:,["DEPTH"]+ssp_select]

# Check for sound ducts for the selected data, ducts[:,0] = 'SLD', ducts[:,1] = 'DC'
ducts = np.zeros([np.size(data,0),3],int)
i = 0
for ssp,dmax,idx in zip(data['profile'],data['water_depth_max'], data.index):
    for SSP,DMAX,SLD,DC in zip(SSP_Prop['SSP'], SSP_Prop['dmax'], SSP_Prop['SLD_depth'], SSP_Prop['DC_axis']):
        ducts[i,0] = idx
        if str(ssp) == str(SSP) and int(dmax) == int(DMAX):
            if np.isnan(SLD)==False:
                ducts[i,1] = 1
            if np.isnan(DC)==False:
                ducts[i,2] = 1

    i += 1
  

Entities = [
    {"NodeName": 'sound-propagation-scenario',
     "QueryList": Scenario(data)
    },
    
    {"NodeName": 'ray-input',
     "QueryList": RayInput(data)
     },
    
    {"NodeName": 'source',
     "QueryList": Source(data)
     },
    
    {"NodeName": 'bottom-segment-ALL',
     "QueryList": BottomSegment_ALL(Bathy)
     },
    
    {"NodeName": 'SSP-vec',
     "QueryList": SSPVec(SSP_Input, SSP_Stat)
     },
    
    {"NodeName": 'sonic-layer',
     "QueryList": SonicLayer(SSP_Prop)
     },
    
    {"NodeName": 'deep-channel',
     "QueryList": DeepChannel(SSP_Prop)
     },
    
    {"NodeName": 'not-a-duct',
     "QueryList": DuctExists()
     }
]

Relations = [
        
    {"NodeName": 'relation: convergence',
     "QueryList": rel_Convergence(data)
     },
    
    {"NodeName": 'relation: src-position',
     "QueryList": rel_SrcPosition(data, Bathy)
     },

    {"NodeName": 'relation: bathymetry',
     "QueryList": rel_Bathymetry(data, Bathy)
     },
    
     {"NodeName": 'relation sound-speed',
     "QueryList": rel_SoundSpeed(data, SSP_Input, SSP_Stat)
     },
     
     {"NodeName": 'relation SSP-channel',
     "QueryList": rel_SSPChannel(SSP_Input, SSP_Stat, SSP_Prop)
     },
     
     {"NodeName": 'relation SSP-value to Depth',
     "QueryList": rel_SSPvecToDepth(SSP_Input)
     }
         
]
"""
Entities = [
    {"NodeName": 'dummy',
     "QueryList": []
     }
]
Relations = [
    {"NodeName": 'dummy',
    "QueryList": []
    },
       {"NodeName": 'relation SSP-vec order',
     "QueryList": rel_SSPvecOrdered(data_complete)
     }
]
"""

if __name__ == "__main__":
    build_graph(Inputs=[Entities, Relations], keyspace_name = "ssp_schema_kgcn") #Entities,
    print("Importing data to GRAKN finished OK!")
    