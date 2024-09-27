
import pandas as pd
import ast
from evaluation import get_known, adjust_capacity_by_failure_rate, get_maintenance_cost
from helpers.datacenters import Datacenter
from helpers.server_type import Server

from waqee import moveLP

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
    selling_prices: list
        A list of selling prices for each server generation at each timestep
    OBJECTIVE: float
        The objective value
    verbose: int
        0 - supress output
        1 - ouput objective value at each timestep
    
    """

    def __init__(self, datacenters, server_types, selling_prices, demand, seed, verbose = 1):
        self.datacenters = dict()
        self.server_types = dict()
        self.seed = seed
        self.p = {2521: 0.15,
                    2381: 0.18,
                    5351: 0.24,
                    6047: 0.2,
                    6829: 0.16,
                    9221: 0.16,
                    9859: 0.21,
                    8053: 0.23,
                    1097: 0.15,
                    8677: 0.2}

        self.k = {2521: 13,
                    2381: 4,
                    5351: 10,
                    6047: 3,
                    6829: 3,
                    9221: 5,
                    9859: 3,
                    8053: 5,
                    1097: 3,
                    8677: 3}
        
        self.setServerTypes(server_types, selling_prices)        
        self.setDataCenters(datacenters)

        self.demand = demand
        self.lookaheads = {2521: 15,
                           2381: 23,
                           5351: 16,
                           6047: 16,
                           6829: 15,
                           9221: 15,
                           9859: 15,
                           8053: 19,
                           1097: 17,
                           8677: 20}

        self.id = 0
        self.timestep = 0   

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
        """Gets the actual current demand

        Returns:
            dict[str, dict[str, int]]: The current demand for each latency sensitivity and each server generation 
        """        
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

        ## CREATE LINEAR PROGRAMMING PROBLEM
        m = moveLP(self.datacenters,
                   self.server_types,
                   current_demand,
                   self.timestep,
                   p,
                    predicted_demand=self.get_real_ahead_demand(lookahead),
                    lifetimes_left=lifetimes_left,                 
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

        ## CALCULATE OBJECTIVE
        self.OBJECTIVE += self.calculateObjective(self.get_actual_demand(),move_ids)

        if self.verbose:
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
