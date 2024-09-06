import math
import time
from typing import List
import pandas as pd
import numpy as np
from evaluation import get_known
from helpers.datacenters import Datacenter
from helpers.server_type import Server

import ast
import linear_programming

from waqee import moveLP

from statsmodels.tsa.api import Holt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


class DecisionMaker(object):




    
    def __init__(self, datacenters, server_types, selling_prices, demand, seed):

        self.datacenters = dict()
        self.server_types = dict()
        self.seed = seed
        self.p = {2311: 0.24,
                    3329: 0.27,
                    4201: 0.17,
                    8761: 0.16,
                    2663: 0.24,
                    4507: 0.22,
                    6247: 0.21,
                    2281: 0.21,
                    4363: 0.2,
                    5693: 0.29}


        self.k = {2311: 32,
                    3329: 32,
                    4201: 19,
                    8761: 15,
                    2663: 28,
                    4507: 19,
                    6247: 17,
                    2281: 26,
                    4363: 17,
                    5693: 15}
        
        self.setServerTypes(server_types, selling_prices)        
        self.setDataCenters(datacenters)

        self.demand = demand
        self.lookaheads = {2311: 30,
                           3329: 15,
                           4201: 22,
                           8761: 30,
                           2663: 21,
                           4507: 15,
                           6247: 16,
                           2281: 27,
                           4363: 30,
                           5693: 30}

        self.id = 0
        self.timestep = 0
        self.canBuy = True
        

        self.solution = []

    def generateUniqueId(self) -> str:
        id = "server-" + str(self.id)
        self.id += 1
        return id
    
    def setDataCenters(self, datacenters) -> None:

        ## create all datacenters
        ## only add a server type to a datacentre, if they have the same latency sensitivity
        self.datacenters = {dc.datacenter_id: Datacenter( dc.datacenter_id,
                                                          dc.cost_of_energy,
                                                          dc.latency_sensitivity, 
                                                          dc.slots_capacity,
                                                            # dict(
                                                            #     filter(lambda item: item[1].latency_sensitivity == dc.latency_sensitivity, self.server_types.items())
                                                            # )
                                                            self.server_types
                                                        ) for dc in datacenters.itertuples()}
    

    def setServerTypes(self, server_types, selling_prices) -> None:

        k = 20 if self.seed not in self.k.keys() else self.k[self.seed]

        self.server_types = {s.server_generation: Server(
                                                                                        s.server_generation,
                                                                                        ast.literal_eval(s.release_time),
                                                                                        s.purchase_price, 
                                                                                        s.slots_size,
                                                                                        s.energy_consumption,
                                                                                        s.capacity,
                                                                                        s.life_expectancy,
                                                                                        s.cost_of_moving,
                                                                                        s.average_maintenance_fee,
                                                                                        k
                                                                                        
                                                                                        ) for s in server_types.itertuples() }



        ## add the selling prices to the server type
        for server in self.server_types.values():
            
            selling_prices_for_server = selling_prices.loc[(selling_prices["server_generation"]==server.name)]

            ## set selling prices for server as a dictionary 
            ## that maps the latency to that servers selling price at that latency 
            server.setSellingPrices( pd.Series( 
                                                selling_prices_for_server.selling_price.values,
                                                index =selling_prices_for_server.latency_sensitivity
                                            ).to_dict()
                                    )
        
        
        
        
        

    

    def buyServers(self, datacenter: Datacenter, server_type: str, quantity: int) -> None:
        # check if server type exists
        if server_type not in self.server_types:
            raise ValueError(f"Server type '{server_type}' does not exist.")
        
        # check if datacenter exists
        if datacenter.name not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter.name}' does not exist.")
        
        
        assert(datacenter.inventory_level + (self.server_types[server_type].slots_size * quantity)
               <= datacenter.slots_capacity)

        for _ in range(quantity):
            server_id = self.generateUniqueId()
            datacenter.buy_server(server_type, server_id, self.timestep)
            
            self.addToSolution(self.timestep, datacenter.name, server_type, server_id, "buy")

    def sellServers(self, datacenter: Datacenter, server_type: str, quantity: int) -> None:
        # check if server type exists
        if server_type not in self.server_types:
            raise ValueError(f"Server type '{server_type}' does not exist.")
        
        # check if datacenter exists
        if datacenter.name not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter.name}' does not exist.")
        
        assert(len(datacenter.inventory[server_type]) >= quantity)
        
        for _ in range(quantity):
            server_id = datacenter.sell_server(server_type)
            self.addToSolution(self.timestep, datacenter.name, server_type, server_id, "dismiss")

    def moveServers(self) -> None:
        sorted_datacenters = sorted(self.datacenters.values(), key= lambda x: x.getProfitability(), reverse=True)
        for buyer in sorted_datacenters:
            for server in [add for add in buyer.adding_servers.keys() 
                           if buyer.adding_servers[add] > 0]:
                for _ in range(buyer.adding_servers[server]):
                    valid_dcs = [dc for dc in sorted_datacenters if buyer!=dc 
                                and dc.removing_servers[server] > 0]
                    
                    if len(valid_dcs) == 0:
                        break

                    seller = max(valid_dcs, key=lambda x: x.inventory[server][-1][0])
                    server_id = buyer.move_server(seller, server)
                    self.addToSolution(self.timestep, buyer.name, server, server_id, "move")
    
    
    def checkConstraints(self) -> None:

        ## check whether a datacenter contains servers that exceed their lifetime 
        for datacenter in self.datacenters.values():
            datacenter.check_lifetime(self.timestep)


    ## adds a transaction to the solution 
    ## returns true if succesful
    def addToSolution(self, timestep: int, datacenter: str, server_type: str, server_id : str,  action: str) -> bool:
        
        transaction = {
                        "time_step": timestep,
                        "datacenter_id": datacenter,
                        "server_generation": server_type,
                        "server_id": server_id,
                        "action": action
                        }
        
     
        
        self.solution.append(transaction)
        return True 
    

    def getLatencyDataCenters(self, latency_sensitivity: str) -> dict[str,Datacenter]:
        return {d: self.datacenters[d] for d in self.datacenters.keys() 
                if self.datacenters[d].latency_sensitivity == latency_sensitivity }
    
    

    """
        returns a nested dictionary mapping datacentres to servers to the lifetime remaining of the oldest server of that type in that datacenter
        e.g. lifetimes_left["DC1"]["CPU.S1"]== 24
    """
    def getLifetimesLeft(self)-> dict[str,dict[str,int]]:
        
        lifetimes_left = {}

        for dc, datacenter in self.datacenters.items():

            dc_lifetimes = {}

            for s, server in self.server_types.items():
                
                if  len(datacenter.inventory[s]) == 0 :
                    dc_lifetimes[s]=0
                    

                else:

                    timestep_bought  = datacenter.inventory[s][0][0]
                    dc_lifetimes[s] = server.life_expectancy - (self.timestep - timestep_bought)


                
            lifetimes_left[dc] = dc_lifetimes
        
        return lifetimes_left



    def getCanBuys(self):

        pass
    
    def getDemandCoeffs(self, datacenters: List[Datacenter]) -> dict[str,float]:
        energy_cost_sum = 0
        remaining_capacity_sum = 0
        for datacenter in datacenters:
            energy_cost_sum += datacenter.cost_of_energy
            remaining_capacity_sum += datacenter.remainingCapacity()
        
        return {d.name: self.calculateCoeff( 
                                                d.cost_of_energy,
                                                d.remainingCapacity(),
                                                energy_cost_sum,
                                                remaining_capacity_sum
                                            )
                                        for d in datacenters
                                }
        

    def calculateCoeff(self, energy_cost: int, remaining_capacity: int, energy_cost_sum: int, 
                       remaining_capacity_sum: int) -> float:
        
        energy_frac = (energy_cost/energy_cost_sum)
        if energy_frac != 1:
            energy_frac = 1 - energy_frac
        return energy_frac
    
    """ 
        returns ALL  of the demand coeffecients as a nested dictionary 
        first dictionary points to the latency_sensitivity of datacenters
        second dictionary is the coefficient for how much of the demand should go to that dictionary 

        e.g. coefficients["low"]   , will return a dictionary which maps "low" datacentres to their respective cofficients
        e.g. coefficients["low"]["DC1"] will return the coefficient for D   
        TODO: it may be worth testing that for coefficients[latency_sensitivity] all the coefficients add up to 1
    """
    def get_all_demand_coefficients(self) -> dict[str, dict[str, float]]:

        demand_coefficients = {}
        for latency_sensitivity in get_known('latency_sensitivity'):
                
            ## gets all datacenters with this latency 
            datacenters_with_latency = [dc for dc in self.datacenters.values() if dc.latency_sensitivity == latency_sensitivity]
                
            demand_coefficients[latency_sensitivity] = self.getDemandCoeffs(datacenters_with_latency) 

        return demand_coefficients
    
    def get_real_ahead_demand(self, lookahead:int) -> dict[str, dict[str, int]]:

        demands = {}

        for latency_sensitivity in get_known('latency_sensitivity'):

                    ## get all data centers for this latency  
                    dcs = self.getLatencyDataCenters(latency_sensitivity)
                    
                    ## dataframe of rows where t = current time step
                    if self.timestep > get_known("time_steps") - (lookahead+1):
                        ts_demand = self.demand.loc[(self.demand['time_step']==self.timestep)].copy()
                    else:
                        ts_demand = self.demand.loc[(self.demand['time_step']==self.timestep+lookahead)].copy()


                    latency_demands = {}

                    ## for all servers
                    for server in self.server_types.keys():
                        
                        server_demand_df = ts_demand.loc[(ts_demand['server_generation']==server) ].copy()
                        if server_demand_df.empty:
                            latency_demands[server] = 0
                        else:
                            latency_demands[server] = int(server_demand_df.iloc[0][latency_sensitivity] * (10/9))

                            

                        

                    demands[latency_sensitivity] = latency_demands

        return demands


    """
     Processes the demand from the csv
    """
    def processDemand(self) -> dict[str, dict[str, int]]:

        demands = {}

        for latency_sensitivity in get_known('latency_sensitivity'):

                    ## get all data centers for this latency  
                    dcs = self.getLatencyDataCenters(latency_sensitivity)
                    
                    ## dataframe of rows where t = current time step
                    ts_demand = self.demand.loc[(self.demand['time_step']==self.timestep)].copy()


                    latency_demands = {}

                    ## for all servers
                    for server in self.server_types.keys():
                        
                        server_demand_df = ts_demand.loc[(ts_demand['server_generation']==server) ].copy()
                        if server_demand_df.empty:
                            latency_demands[server] = 0
                        else:
                            ## demand for the server generation at this latency for all timesteps <= current timestep
                            server_demand =  self.demand.loc[(self.demand['server_generation'] == server)
                                                             & (self.demand['time_step'] <= self.timestep)].copy()
                            ls_demand = server_demand[['time_step', latency_sensitivity]].copy()
                            endog = ls_demand[latency_sensitivity].to_numpy()
                            
                            ## If we have demand for only one timestep, use the actual demand
                            if len(endog)==1:
                                latency_demands[server] = int(server_demand_df.iloc[0][latency_sensitivity] * (10/9))
                                continue
                            
                            ## Apply holt's damped smoothing to the demand
                            np.seterr(divide='ignore')
                            fit = Holt(endog, damped_trend=True, initialization_method="estimated").fit(
                                    smoothing_level=0.2, smoothing_trend=0.12, 
                                )
                            d = fit.fittedvalues

                            ## In some cases such as holt's, it will produce negative values, so just set it to 0
                            ## TODO: Deal with this in a better way. boxcox parameter in Holt can potentially be used
                            d[d<0] = 0
                            f = fit.forecast(1)
                            f[f<0] = 0
                            latency_demands[server] = int(np.average(f) * (10/9))
                            np.seterr(divide='warn')

                    demands[latency_sensitivity] = latency_demands

        return demands

    ## returns a dictionary that maps latencys to datacenters to demand for each server
    ## e.g. demand["high"]["DC3"]["CPU1"] = 99
    ## equals the demand for high latency_sensitivity CP1 at DC1
    def getWeightedDemand(self, cur_demand: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:

        ## data center capacity score
        ## we should score each data centre for a particular latency based on how much capacity it has remaining 
        
        ## based on the data center capcity score,
        ## we should assign a certain fraction of the demand to it 
        ## so if DC3 has a score of 30 and DC4 has a score of 10
        ## DC3 should take 0.75 of the capacity of the demand 

        ## we then rank the relevant datacenters by cost ascending, e.g. [DC4, DC3]    --   DC4.cost < DC3.cost
        ## then assign DC4 the most expensive server until it has been assigned 0.25 of the capacity of the demand
        ## carry on until all the demand for all servers has been assigned a datacenter 

        ## Hopefully this would mean that datacenters with lower energy costs are given more expensive servers to run 

        weighted_demand = dict()
        for latency_sensitivity in get_known('latency_sensitivity'):

            # Get all datacenters from a latency sensitivity
            dcs = list(self.getLatencyDataCenters(latency_sensitivity).values())

            # There is only 1 datacenter in a latency sensitivity so there is no need
            # apply weights
            if len(dcs) == 1:
                weighted_demand[dcs[0].name] = cur_demand[latency_sensitivity]
                continue
            remaining_capacity_sum = sum([dc.remainingCapacity() for dc in dcs])

            # Calculate the fraction of demand each datacenter should have based on the
            # remaining capacity
            dc_coeffs = {dc.name: (dc.remainingCapacity()/remaining_capacity_sum if 
                                  remaining_capacity_sum > 0 else 0) for dc in dcs}

            # Rank datacenters based on cost of energy in ascending order
            dc_rank = sorted(dcs, key=lambda x: x.cost_of_energy)
            latency_demand = cur_demand[latency_sensitivity]

            
            for dc in dc_rank:

                met_demand = 0
                demand_coeff = dc_coeffs[dc.name]
                total_latency_demand = sum(latency_demand.values())
                demand_to_meet = total_latency_demand * demand_coeff
                server_rank_by_cost = sorted(self.server_types.values(), key=lambda x: x.energy_consumption, reverse=True)
                dc_demand = dict()
                
                # Assign the highest cost servers to the lowest cost datacenter
                for server in server_rank_by_cost:
                    server_demand = latency_demand[server.name]
                    dc_demand[server.name] = 0
                    if latency_demand[server.name] == 0:
                        continue

                    for _ in range(server_demand):
                        if met_demand + self.server_types[server.name].capacity > demand_to_meet:
                            break
                        dc_demand[server.name] += 1
                        latency_demand[server.name] -= 1
                weighted_demand[dc.name] = dc_demand
        return weighted_demand

    def step(self):
        self.timestep += 1
        print(self.timestep)

        if self.timestep >= get_known('time_steps')-35:
            self.canBuy = False

        
        
        self.checkConstraints()


        ## PROCESS DEMAND FROM CSV
        current_demand = self.get_real_ahead_demand(0)##self.processDemand()
        demand_coeffs = self.get_all_demand_coefficients()
        #weighted_demand = self.getWeightedDemand(current_demand)

        lifetimes_left = self.getLifetimesLeft()

        lookahead = 20 if self.seed not in self.lookaheads.keys() else self.lookaheads[self.seed]
        
        p = 0.2 if self.seed not in self.p.keys() else self.p[self.seed]
        m = moveLP(self.datacenters,
                   self.server_types,
                   current_demand,
                   self.timestep,
                   p,
                    predicted_demand=self.get_real_ahead_demand(lookahead),
                    lifetimes_left=lifetimes_left,
                    can_buy= self.canBuy,
                    
                )
        m.solve()

        assert m.model.status == 1, f"LINEAR PROGRAMMING PROBLEM HAS NOT FOUND A FEASIBLE SOLUTION: STATUS CODE = {m.model.status}"
        
        # ## GET NUMBER OF ADD AND REMOVE FOR EACH DATACENTRE 
        # for datacenter in self.datacenters.values():

        #     weighted_demand = { server:
        #                                  current_demand[datacenter.latency_sensitivity][server]
        #                                  * 
        #                                  demand_coeffs[datacenter.latency_sensitivity][datacenter.name]
        #                         for server in self.server_types.keys()
        #                     }
        #     datacenter.find_add_remove_for_all_servers(timestep=self.timestep,
        #                                                 demands = weighted_demand
        #                                                 )
        
        # self.moveServers()

        adds, removes, moves = 0,0,0 

        for moveVar in m.variables:
            details = moveVar.split("_")
            from_dc = details[0]
            to_dc = details[1]
            s = details[2]

            moves += m.variables[moveVar].varValue

            for _ in range(int(m.variables[moveVar].varValue)):
                server_id = self.datacenters[to_dc].move_server(self.datacenters[from_dc], s)
                self.addToSolution(self.timestep, to_dc, s, server_id, "move")

        for removeVar in m.removeVariables:
            details = removeVar.split("_")
            dc = details[0]
            s = details[1]

            removes += m.removeVariables[removeVar].varValue

            self.sellServers(self.datacenters[dc], s, int(m.removeVariables[removeVar].varValue))
        
        # if self.canBuy:
        for addVar in m.addVariables:
            details = addVar.split("_")
            dc = details[0]
            s = details[1]

            adds += m.addVariables[addVar].varValue

            # ts_demand = self.demand.loc[(self.demand['time_step']==self.timestep)
            #                             & (self.demand['server_generation']==s)].copy()
            # s_demand = 0
            # if not ts_demand.empty:
            #     s_demand = (ts_demand.iloc[0][self.datacenters[dc].latency_sensitivity] * 
            #                 demand_coeffs[self.datacenters[dc].latency_sensitivity][self.datacenters[dc].name])
            
            # actual = int((s_demand - 
            #               (len(self.datacenters[dc].inventory[s]) * self.server_types[s].capacity))
            #               //self.server_types[s].capacity)

            # if actual < 0:
            #     actual = 0
            #self.buyServers(self.datacenters[dc], s, min(actual, int(m.addVariables[addVar].varValue)))
            self.buyServers(self.datacenters[dc], s, int(m.addVariables[addVar].varValue))

        
        # ## CARRY OUT TRANSACTIONS LIKE BUY, DISMISS, MOVE
        # for datacenter in self.datacenters.values():

        #     for server, remove_amount in datacenter.removing_servers.items():
                        
        #             self.sellServers(datacenter,server,remove_amount)

        #     if self.canBuy:
        #         for server, add_amount in datacenter.adding_servers.items():
                            
        #                 self.buyServers(datacenter,server,add_amount)

           
        print(f"ADDS: {adds}  REMOVES: {removes}  MOVES: {moves}")






    def solve(self):

        
        for t in range(1, get_known('time_steps')+1):
            self.step()


        return self.solution




   
   
   



    















    
    #calculate L, the normalized lifespan
    def calculate_lifespan(datacenter):
        total_servers = 0
        lifespan_sum = 0

        for server_type, server in datacenter.server_types.items():
            servers = datacenter.inventory[server_type]

            for deployed_time in servers:
                total_servers += 1
                
                #add the ratio of the server's operating time to its life expectancy to sum
                lifespan_sum += deployed_time / server.life_expectancy

        #calculate the normalized lifespan (L) as the average ratio across all servers
        return lifespan_sum / total_servers if total_servers > 0 else 0 # If there are no servers, return 0 to avoid division by zero
    

    #calculate U
    def calculate_utilization(datacenter, demand, failure_rate):
        total_pairs = 0
        utilization_sum = 0

        for (latency_sensitivity, server_generation), demand_values in demand.items():
            
            server_type = datacenter.server_types[server_generation]
            capacity = server_type.capacity

            #adjust capacity for server failure rate
            adjusted_capacity = (1 - failure_rate) * capacity

            #calculate utilization for each time step in the demand values
            for demand_value in demand_values:
                met_demand = min(adjusted_capacity, demand_value)

                utilization_ratio = met_demand / adjusted_capacity
                utilization_sum += utilization_ratio
                total_pairs += 1

        # calculate average utilization U
        return utilization_sum / total_pairs if total_pairs > 0 else 0 # If there are no pairs, return 0 to avoid division by zero.


    #calculate U at timestep
    def calculateUtilizationAtTimestep(self, timestep: int) -> float:
        utilization_sum = 0
        total_pairs = 0

        #get the current demand
        current_demand = self.processDemand()

        for datacenter in self.datacenters.values():
            #get demand coefficients
            demand_coeffs = self.getDemandCoeffs([datacenter])
            #get weighted demand for the datacenter
            weighted_demand = {
                server: current_demand[datacenter.latency_sensitivity][server] * demand_coeffs[datacenter.name]
                for server in self.server_types.keys()
            }

            for server, demand_value in weighted_demand.items():
                #get server capacity
                capacity = datacenter.getServerCapacity(server)
                adjusted_capacity = capacity * (1 - get_known.adjust_capacity_by_failure_rate())
                met_demand = min(adjusted_capacity, demand_value)

                utilization_sum += met_demand / adjusted_capacity
                total_pairs += 1

        return utilization_sum / total_pairs if total_pairs > 0 else 0  # If there are no pairs, return 0 to avoid division by zero.


    #calculate the lifespan at a given timestep
    def calculateLifespanAtTimestep(self, timestep: int) -> float:
        lifespan_sum = 0
        total_servers = 0

        for datacenter in self.datacenters.values():
            for server_type, server_list in datacenter.inventory.items():
                for deployed_time in server_list:
                    server = self.server_types[server_type]
                    lifespan_sum += deployed_time / server.life_expectancy
                    total_servers += 1

        return lifespan_sum / total_servers if total_servers > 0 else 0 # If there are no pairs, return 0 to avoid division by zero.
    

    #calculate the profit at a given timestep
    def calculateProfitAtTimestep(self, servers: List[Server], timestep: int) -> float:
        profit = 0

        for datacenter in self.datacenters.values():
            revenue = 0
            costs = 0

            for server_type, server_list in datacenter.inventory.items():
                server = self.server_types[server_type]
                for server_id, deployed_time in server_list:
                    #revenue =  selling price * capacity * latency sensitivity
                    revenue += server.selling_prices[datacenter.latency_sensitivity] * server.capacity

                    #costs = purchase price, maintenance fee, and energy cost
                    costs += server.purchase_price
                    #FIXME: cost of maintenance is ignored for now
                    # costs += server.average_maintenance_fee
                    costs += datacenter.cost_of_energy * server.energy_consumption

                    # linear_programming.purchase_price(datacenter.latency_sensitivity)

            profit += (revenue - costs)

        return profit

    # Calculate the objective value.
    def calculateObjectiveAtTimestep(self, timestep: int) -> float:
        U = self.calculateUtilizationAtTimestep(timestep)
        L = self.calculateLifespanAtTimestep(timestep)
        P = self.calculateProfitAtTimestep(timestep)

        #FIXME: is there a weight for each of these?
        # utilizationWeight = 1.0
        # lifespanWeight = 1.0
        # profitWeight = 1.0

        if U <= 0 or L <= 0 or P <= 0:
            return 0

        objective_value = U*L*P
        # objective_value = utilizationWeight * U + lifespanWeight * L + profitWeight * P
        return objective_value


