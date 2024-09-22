from typing import List
import pandas as pd
import numpy as np
import ast
from evaluation import get_known, adjust_capacity_by_failure_rate, get_maintenance_cost
from helpers.datacenters import Datacenter
from helpers.server_type import Server

from waqee import moveLP

from statsmodels.tsa.api import Holt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)



class DecisionMaker(object):
    """
    A class to represent the decision maker.

    Attributes
    ----------
    datacenters : Dict
        A dictionary of all datacenters
    server_types : Dict
        A dictionary of all server generations
    seed : int
        The random seed that is being used 
    p : dict[int, float]
        The optimal profit parameter used in elastic constraint for each seed
    buyOnce : dict[int, bool]
        Whether each seed should buy every timestep (True) or every other timestep (False)
    k : dict[int, int]
        The optimal k parameter in server_type.getTimeTillProfitable for each seed
    demand : pd.DataFrame
        A pandas dataframe of demand for each latency sensitivity and each server generation for all timesteps
    lookaheads : dict[int, int]
        The optimal value to look into the future demand for each seed
    id : int
        The next unique server id
    timestep : int
        The current timestep
    solution : list
        A list of transactions
    
    """



    def __init__(self, datacenters, server_types, selling_prices, demand, seed, verbose = 1):
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
        
        self.buyOnce = {2311: False,
                    3329: True,
                    4201: False,
                    8761: False,
                    2663: False,
                    4507: True,
                    6247: False,
                    2281: False,
                    4363: False,
                    5693: False}

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
        self.sellingPrices = []

        self.OBJECTIVE = 0 
        self.verbose = verbose

    def generateUniqueId(self) -> str:
        """Generates a unique server id

        Returns:
            str: A unique server id
        """
        id = "server-" + str(self.id)
        self.id += 1
        return id
    
    def setDataCenters(self, datacenters: pd.DataFrame) -> None:
        """Sets the datacenters attribute 

        Args:
            datacenters (pd.DataFrame): A pandas dataframe containing all datacenters
        """
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
    
    def setServerTypes(self, server_types: pd.DataFrame, selling_prices: pd.DataFrame) -> None:
        """Sets the server types attribute

        Args:
            server_types (pd.DataFrame): A pandas dataframe containing all server generations
            selling_prices (pd.DataFrame): A pandas dataframe containing the selling prices for each server generation
                                            and latency sensitivity
        """

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
        """Buys a number of servers for a datacenter

        Args:
            datacenter (Datacenter): The datacenter to add servers to
            server_type (str): The server generation that we are buying
            quantity (int): The number of servers to buy

        Raises:
            ValueError: server_type is not a valid server generation
            ValueError: datacenter does not exist
        """
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
        """Removes a number of servers from a datacenter

        Args:
            datacenter (Datacenter): The datacenter that is being removed from
            server_type (str): The server generation that is being removed
            quantity (int): The number of servers to remove

        Raises:
            ValueError: server_type is not a valid server generation
            ValueError: datacenter does not exist
        """
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

    @DeprecationWarning
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
        """Check whether a datacenter contains servers that exceed their lifetime 
        """

        for datacenter in self.datacenters.values():
            datacenter.check_lifetime(self.timestep)

    def addToSolution(self, timestep: int, datacenter: str, server_type: str, server_id : str,  action: str) -> bool:
        """Add a transaction to the solution 

        Args:
            timestep (int): The timestep the transaction is taking place
            datacenter (str): The datacenter the transaction is taking place at
            server_type (str): The server generation
            server_id (str): The id of the server
            action (str): What transaction we are taking (buy/dismiss/move)

        Returns:
            bool: True if the transaction was successful
        """
        transaction = {
                        "time_step": timestep,
                        "datacenter_id": datacenter,
                        "server_generation": server_type,
                        "server_id": server_id,
                        "action": action
                        }

        price = {
            "time_step": timestep,
            "latency_sensitivity": self.datacenters[datacenter].latency_sensitivity,
            "server_generation": server_type,
            "price": self.server_types[server_type].selling_prices[self.datacenters[datacenter].latency_sensitivity]
        }
        
        self.solution.append(transaction)
        self.sellingPrices.append(price)
        
        return True   

    def getLatencyDataCenters(self, latency_sensitivity: str) -> dict[str,Datacenter]:
        """Gets all datacenters of a latency sensitivity

        Args:
            latency_sensitivity (str): The latency sensitivity to filter for

        Returns:
            dict[str,Datacenter]: A dictionary of datacenters that has the required latency.
        """        
        return {d: self.datacenters[d] for d in self.datacenters.keys() 
                if self.datacenters[d].latency_sensitivity == latency_sensitivity }
    
    def getLifetimesLeft(self)-> dict[str,dict[str,int]]:
        """Gets the remaining lifetime of the oldest server at each datacenter

        Returns:
            dict[str,dict[str,int]]: a nested dictionary mapping datacentres to servers to the lifetime remaining 
                                        of the oldest server of that type in that datacenter, e.g. lifetimes_left["DC1"]["CPU.S1"]== 24
        """        
        
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

    @DeprecationWarning
    def getCanBuys(self):
        pass
    
    @DeprecationWarning
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

    @DeprecationWarning    
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
    @DeprecationWarning
    def get_all_demand_coefficients(self) -> dict[str, dict[str, float]]:

        demand_coefficients = {}
        for latency_sensitivity in get_known('latency_sensitivity'):
                
            ## gets all datacenters with this latency 
            datacenters_with_latency = [dc for dc in self.datacenters.values() if dc.latency_sensitivity == latency_sensitivity]
                
            demand_coefficients[latency_sensitivity] = self.getDemandCoeffs(datacenters_with_latency) 

        return demand_coefficients
    
    def get_real_ahead_demand(self, lookahead:int) -> dict[str, dict[str, int]]:
        """Gets the actual future demand

        Args:
            lookahead (int): The number of timesteps to lookahead

        Returns:
            dict[str, dict[str, int]]: The future demand for each latency sensitivity and each server generation 
        """        

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

    def get_actual_demand(self) -> dict[str, dict[str, float]]:
        demands = {}

        for latency_sensitivity in get_known('latency_sensitivity'):
            latency_demands = {}
            ts_demand = self.demand.loc[(self.demand['time_step']==self.timestep)].copy()

            for server in self.server_types.keys():
                server_demand_df = ts_demand.loc[(ts_demand['server_generation']==server) ].copy()
                if server_demand_df.empty:
                    latency_demands[server] = 0
                else:
                    latency_demands[server] = server_demand_df.iloc[0][latency_sensitivity]
            demands[latency_sensitivity] = latency_demands

        return demands

    """
     Processes the demand from the csv
    """
    @DeprecationWarning
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
    @DeprecationWarning
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
        

        if self.timestep >= get_known('time_steps')-35:
            self.canBuy = False
   
        self.checkConstraints()

        ## PROCESS DEMAND FROM CSV
        current_demand = self.get_real_ahead_demand(0)
        

        lifetimes_left = self.getLifetimesLeft()

        lookahead = 20 if self.seed not in self.lookaheads.keys() else self.lookaheads[self.seed]
        
        p = 0.2 if self.seed not in self.p.keys() else self.p[self.seed]
        buyOnce = False if self.seed not in self.buyOnce.keys() else self.buyOnce[self.seed]


        ## CREATE LINEAR PROGRAMMING PROBLEM
        m = moveLP(self.datacenters,
                   self.server_types,
                   current_demand,
                   self.timestep,
                   p,
                   buyOnce,
                    predicted_demand=self.get_real_ahead_demand(lookahead),
                    lifetimes_left=lifetimes_left,
                    can_buy= self.canBuy,                  
                )
        
        ## SOLVE LINEAR PROGRAMMING PROBLEM
        m.solve()

        assert m.model.status == 1, f"LINEAR PROGRAMMING PROBLEM HAS NOT FOUND A FEASIBLE SOLUTION: STATUS CODE = {m.model.status}"
        
      
        
        ## MOVE SERVERS
        move_ids = set()
        for moveVar in m.moveVariables:
            
            details = moveVar.split("_")
            from_dc = details[0]
            to_dc = details[1]
            s = details[2]

            
            for _ in range(int(m.moveVariables[moveVar].varValue)):
                server_id = self.datacenters[to_dc].move_server(self.datacenters[from_dc], s)
                move_ids.add(server_id)
                self.addToSolution(self.timestep, to_dc, s, server_id, "move")



        ## REMOVE SERVERS
        for removeVar in m.removeVariables:
            
            details = removeVar.split("_")
            dc = details[0]
            s = details[1]

            self.sellServers(self.datacenters[dc], s, int(m.removeVariables[removeVar].varValue))
        



        ## ADD SERVERS
        for addVar in m.addVariables:
            
            details = addVar.split("_")
            dc = details[0]
            s = details[1]

            
            self.buyServers(self.datacenters[dc], s, int(m.addVariables[addVar].varValue))

        
     
        if self.verbose:
            ## CALCULATE OBJECTIVE
            self.OBJECTIVE += self.calculateObjective(self.get_actual_demand(),move_ids)

            print(f"{self.timestep}  OBJ: { self.OBJECTIVE}" )



    def solve(self):

        for t in range(1, get_known('time_steps')+1):
            self.step()
        
        return self.solution, self.sellingPrices, self.OBJECTIVE
    

    
    



    def getCapacityDeployed(self, server_type, latency):
        """
            Returns the capacity deployed for a particular server type at a particular latency
        """
        
        total_capacity = 0 

        for dc in self.datacenters.values():

            if dc.latency_sensitivity == latency:
                
                total_capacity += len(dc.inventory[server_type]) * self.server_types[server_type].capacity

        return total_capacity
    


    
 

    #calculate U at timestep
    def calculateUtilization(self,demand) -> float:


        u = []

        ## get ratio for each latency and server type
        for latency in get_known("latency_sensitivity"):
            for server_type in self.server_types.values():

                capacity_adjusted_by_failure_rate = adjust_capacity_by_failure_rate(self.getCapacityDeployed(server_type.name,latency))

                demand_at_latency_server =  demand[latency][server_type.name]


                if capacity_adjusted_by_failure_rate > 0 and demand_at_latency_server > 0:
                    
                    u.append(
                                min(capacity_adjusted_by_failure_rate, demand_at_latency_server)
                                /
                                capacity_adjusted_by_failure_rate
                            )
                elif capacity_adjusted_by_failure_rate > 0 and demand_at_latency_server == 0:
                    
                    u.append(0)


        return sum(u)/len(u) if u else 0 

        

    #calculate the lifespan at a given timestep
    def calculateLifespan(self) -> float:
        """
            Gets the L Objective at each timestep

            Returns: L Objective at current timestep
        """

        ## sum of time_deployed/life_expectancy for each server 
        lifespan_sum = 0

        total_servers = 0

        for datacenter in self.datacenters.values():
            for server_type, server_list in datacenter.inventory.items():

                for deployed_time, server_id in server_list:
                    
                    
                    server = self.server_types[server_type]

                    lifespan_sum += (self.timestep - deployed_time + 1) / server.life_expectancy

                    total_servers += 1

        # If there are no pairs, return 0 to avoid division by zero.
        return lifespan_sum / total_servers if total_servers > 0 else 0 
    


    #calculate the profit at a given timestep
    def calculateProfit(self, demand, move_ids:set[str] ) -> float:
        
        total_revenue = 0
        total_costs = 0

        for latency in get_known("latency_sensitivity"):
            for server in self.server_types.values():
                

                ## revenue for a particular server 
                ## at a particular latency is 
                ## the minimum of the demand  and capacity_deployed * failure_rate 
                ## multiplied by the selling price
                revenue = ( 
                            min(
                                adjust_capacity_by_failure_rate(self.getCapacityDeployed(server.name, latency)) ,
                                demand[latency][server.name] 
                            )
                            * 
                            server.selling_prices[latency]
                        )

                total_revenue += revenue


        for dc in self.datacenters.values():
            for server_type in self.server_types.values():

                for server_spawn_time , id in dc.inventory[server_type.name]:
                    
                    
                    ## code taken from evaluation.py, see formula in pdf 
                    c = 0
                    r = server_type.purchase_price
                    b = server_type.avg_maintenance_fee
                    x = self.timestep-server_spawn_time+1 
                    xhat = server_type.life_expectancy 
                    e = server_type.energy_consumption * dc.cost_of_energy 
                    c += e
                    alpha_x = get_maintenance_cost(b, x, xhat)
                    c += alpha_x

                    if x == 1:
                        c += r
                    elif id in move_ids:
                        c += server_type.cost_of_moving
                    

                    total_costs+= c
            

        
        return total_revenue - total_costs




            
       

    # Calculate the objective value. 
    def calculateObjective(self, demand, move_ids) -> float:
        
        # U = self.calculateUtilization(demand)
        # L = self.calculateLifespan()
        P = self.calculateProfit(demand, move_ids)

        
        objective_value = P

        return objective_value
