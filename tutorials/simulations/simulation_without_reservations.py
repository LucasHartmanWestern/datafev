# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:28:41 2022

@author: egu
"""

from datetime import datetime,timedelta
from pyomo.environ import SolverFactory

from datahandling.fleet import EVFleet
from datahandling.cluster import ChargerCluster
from datahandling.multicluster import MultiClusterSystem

from protocols.arrival import *
from protocols.departure import *
from protocols.chargingcontrol.decentralized_fcfs import charging_protocol as charging_protocol_fcs
from protocols.chargingcontrol.decentralized_llf import charging_protocol as charging_protocol_llf

#Simulation parameters
sim_start       =datetime(2022,1,8,7)
sim_end         =datetime(2022,1,8,20)
sim_length      =sim_end-sim_start
sim_step        =timedelta(minutes=5)
sim_horizon     =[sim_start+t*sim_step for t in range(int(sim_length/sim_step))]
solver=SolverFactory("cplex")

#Simulation inputs
inputs          = pd.ExcelFile('scenario_without_resevation.xlsx')
input_fleet     = pd.read_excel(inputs, 'Fleet')
input_cluster1  = pd.read_excel(inputs, 'Cluster1')
input_capacity1 = pd.read_excel(inputs, 'Capacity1')
input_cluster2  = pd.read_excel(inputs, 'Cluster2')
input_capacity2 = pd.read_excel(inputs, 'Capacity2')
input_cluster3  = pd.read_excel(inputs, 'Cluster3')
input_capacity3 = pd.read_excel(inputs, 'Capacity3')

price           = pd.read_excel(inputs, 'Price')
price_t_steps   = price['TimeStep'].round('S')
tou_tariff      = pd.Series(price['Price'].values,index=price_t_steps)


for charging_control in ['uncontrolled','fcfs','llf']:

    print("Simulating the charging control approach:",charging_control)

    #######################################################################
    #Multicluster charging system and EV fleet
    cluster1  = ChargerCluster("cluster1",input_cluster1)
    cluster2  = ChargerCluster("cluster2",input_cluster2)
    cluster3  = ChargerCluster("cluster3",input_cluster3)
    system    = MultiClusterSystem("multicluster")
    system.add_cc(cluster1)
    system.add_cc(cluster2)
    system.add_cc(cluster3)
    system.set_tou_price(tou_tariff,sim_step)

    fleet = EVFleet("test_fleet",input_fleet,sim_horizon)
    #######################################################################

    #######################################################################
    #Additional parameters for charging management protocol
    cluster1.set_peak_limits(sim_start,sim_end,sim_step,input_capacity1)
    cluster2.set_peak_limits(sim_start,sim_end,sim_step,input_capacity2)
    cluster3.set_peak_limits(sim_start,sim_end,sim_step,input_capacity3)
    #######################################################################

    #######################################################################
    #Simulation starts

    np.random.seed(0)

    for ts in sim_horizon:
        print("Simulating time step:",ts)

        #The departure protocol for the EVs leaving the chargerg clusters
        departure_protocol(ts,fleet)

        #The arrival protocol for the EVs incoming to the charger clusters
        arrival_protocol(ts,sim_step,fleet,system)

        #Real-time charging control of the charger clusters
        if charging_control=='uncontrolled':
            system.uncontrolled_supply(ts, sim_step)
        if charging_control=='fcfs':
            charging_protocol_fcs(ts, sim_step,system)
        if charging_control=='llf':
            charging_protocol_llf(ts, sim_step,system)

    #Simulation ends
    #######################################################################

    #######################################################################
    # Printing the results to excel files
    system.export_results(sim_start,sim_end,sim_step,'result_noreservation_'+charging_control+'_clusters.xlsx')
    fleet.export_results(sim_start,sim_end,sim_step,'result_noreservation_'+charging_control+'_fleet.xlsx')
    #######################################################################

#TODO: Add visual outputs