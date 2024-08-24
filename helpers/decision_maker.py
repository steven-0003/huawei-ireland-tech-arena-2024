from helpers.datacenters import Datacenter
from helpers.server_type import Server

import ast
import linear_programming


class DecisionMaker(object):
    def __init__(self, datacenters, server_types, selling_prices):

        self.datacenters = dict()
        self.server_types = dict()

        ## create all server types, a server type is a cpu generation with a specific latency sensitivity 
        self.server_types = {s.server_generation+"_"+latency_sensitivity: Server(
                                                                                s.server_generation+"_"+latency_sensitivity,
                                                                                ast.literal_eval(s.release_time),
                                                                                s.purchase_price, 
                                                                                s.slots_size,
                                                                                s.energy_consumption,
                                                                                s.capacity,
                                                                                s.life_expectancy,
                                                                                s.cost_of_moving,
                                                                                s.average_maintenance_fee,
                                                                                latency_sensitivity
                                                                                ) for s in server_types.itertuples() for latency_sensitivity in ["low","medium","high"]}

        ## add the selling price to the server type
        for s in selling_prices.itertuples():
            self.server_types[s.server_generation+"_"+s.latency_sensitivity].setSellingPrice(s.selling_price)

        ## create all datacenters
        ## only add a server type to a datacentre, if they have the same latency sensitivity
        self.datacenters = {dc.datacenter_id: Datacenter( dc.datacenter_id,
                                                          dc.cost_of_energy,
                                                          dc.latency_sensitivity, 
                                                          dc.slots_capacity,
                                                            dict(
                                                                filter(lambda item: item[1].latency_sensitivity == dc.latency_sensitivity, self.server_types.items())
                                                            )) for dc in datacenters.itertuples()}
    
        self.active_server_types = []

        self.id = 0
        self.timestep = 0
        self.solution = []

    def generateUniqueId(self) -> str:
        id = "server-" + str(self.id)
        self.id += 1
        return id
    
    def addDataCenters(self, datacenters: list[Datacenter]) -> None:
        for datacenter in datacenters:
            self.datacenters[datacenter.name] = datacenter

    def addServerTypes(self, server_types: list[Server]) -> None:
        for server_type in server_types:
            self.server_types[server_type.name] = server_type

    def step(self):
        self.timestep += 1

        self.getActiveServers()
        for datacenter in self.datacenters.keys():
            self.checkConstraints(self.datacenters[datacenter])

    def buyServer(self, datacenter: str, server_type: str) -> None:
        # check if server type exists
        if server_type not in self.server_types:
            raise ValueError(f"Server type '{server_type}' does not exist.")
        
        # check if datacenter exists
        if datacenter not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter}' does not exist.")
        
        assert(self.datacenters[datacenter].inventory_level + self.server_types[server_type].slots_size
               <= self.datacenters[datacenter].slots_capacity)
        
        # generate a unique ID for the new server(unless theres a id already generated/provided)
        server_id = self.generateUniqueId()
        
        # buy server by calling the buy_server method from datacenter.py
        self.datacenters[datacenter].buy_server(server_type, server_id, self.timestep)
        self.solution.append(self.addToSolution(self.timestep, datacenter, server_type, server_id, "buy"))

    def sellServer(self, datacenter: str, server_type: str) -> None:
        # check if server type exists
        if server_type not in self.server_types:
            raise ValueError(f"Server type '{server_type}' does not exist.")
        
        # check if datacenter exists
        if datacenter not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter}' does not exist.")
        
        assert(len(self.datacenters[datacenter].inventory[server_type]) >= 1)
        
        server_id = self.datacenters[datacenter].sell_server(server_type)
        self.solution.append(self.addToSolution(self.timestep, datacenter, server_type, server_id, "dismiss"))

    def buyServers(self, datacenter: str, server_type: str, quantity: int) -> None:
        # check if server type exists
        if server_type not in self.server_types:
            raise ValueError(f"Server type '{server_type}' does not exist.")
        
        # check if datacenter exists
        if datacenter not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter}' does not exist.")
        
        assert(quantity>0)
        assert(self.datacenters[datacenter].inventory_level + (self.server_types[server_type].slots_size * quantity)
               <= self.datacenters[datacenter].slots_capacity)

        for _ in range(quantity):
            server_id = self.generateUniqueId()
            self.datacenters[datacenter].buy_server(server_type, server_id, self.timestep)
            
            self.solution.append(self.addToSolution(self.timestep, datacenter, server_type, server_id, "buy"))

    def sellServers(self, datacenter: str, server_type: str, quantity: int) -> None:
        # check if server type exists
        if server_type not in self.server_types:
            raise ValueError(f"Server type '{server_type}' does not exist.")
        
        # check if datacenter exists
        if datacenter not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter}' does not exist.")
        
        assert(len(self.datacenters[datacenter].inventory[server_type]) >= quantity)
        
        for _ in range(quantity):
            server_id = self.datacenters[datacenter].sell_server(server_type)
            self.solution.append(self.addToSolution(self.timestep, datacenter, server_type, server_id, "dismiss"))

    def checkConstraints(self, datacenter: Datacenter) -> None:
        datacenter.check_lifetime(self.timestep)

    def addToSolution(self, timestep: int, datacenter: str, server_type: str, server_id: str, action: str
                      ) -> dict:
        return {"time_step": timestep, "datacenter_id": datacenter, "server_generation": server_type.split('_')[0], "server_id": server_id, "action": action}
    
    def getLatencyDataCenters(self, latency_sensitivity: str) -> dict[str,Datacenter]:
        return {d: self.datacenters[d] for d in self.datacenters 
                if self.datacenters[d].latency_sensitivity == latency_sensitivity }
    
    def getDemandCoeffs(self, datacenters: dict[str,Datacenter]) -> dict[str,float]:
        energy_cost_sum = 0
        remaining_capacity_sum = 0
        for datacenter in datacenters.keys():
            energy_cost_sum += datacenters[datacenter].cost_of_energy
            remaining_capacity_sum += datacenters[datacenter].remainingCapacity()
        
        return {d: self.calculateCoeff(datacenters[d].cost_of_energy, datacenters[d].remainingCapacity(),
                                       energy_cost_sum, remaining_capacity_sum) for d in datacenters.keys()}
        

    def calculateCoeff(self, energy_cost: int, remaining_capacity: int, energy_cost_sum: int, 
                       remaining_capacity_sum: int) -> float:
        if remaining_capacity_sum == 0:
            return 1/2 * (energy_cost/energy_cost_sum)
        return 1/2 * ((energy_cost/energy_cost_sum) + (remaining_capacity/remaining_capacity_sum))
    
    def getActiveServers(self) -> None:
        self.active_server_types = [server for server in self.server_types.keys() if self.server_types[server].canBeDeployed(self.timestep)]

    # Helper function to extract relevant server data
    def extractRelevantData(self, datacenter_name: str, server_demands: dict[str, int], ls: str, coeff: float) -> list[int]:
        # check if datacenter exists
        if datacenter_name not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter_name}' does not exist.")
        
        # make sure demands are in the same order as the server types (dont know if we need this)
        demands_list = [int(server_demands[server_type_name]*coeff) for server_type_name in self.server_types.keys()
                        if server_type_name.split("_")[1] == ls]

        return demands_list
    
    def getAddRemove(self, demands: list[int], datacenter: str, ls: str):
        servers = [s for s in self.server_types.values() if s.latency_sensitivity == ls]
        servers.sort(key = lambda x: x.name )
        actives = [server.canBeDeployed(self.timestep) for server in servers]

        num_active = len([a for a in actives if a ==True ])

        remaining_slots = self.datacenters[datacenter].remainingCapacity()
        current_server_stock = [len(self.datacenters[datacenter].inventory.get(server_type.name, [])) 
                                for server_type in servers]

        server_capacities = [ s.capacity for s in servers ]

        inequality_matrix = linear_programming.create_inequality_matrix(servers, actives )

        inequality_vector = linear_programming.create_inequality_vector( 
                                                                         remaining_slots,
                                                                         demands,
                                                                         current_server_stock,
                                                                         server_capacities
                                                                        )
        
        objective_vector = linear_programming.create_objective_vector(  servers,
                                                                        actives,
                                                                        self.datacenters[datacenter].cost_of_energy
                                                                    )
        decision_variables = linear_programming.find_add_and_evict(inequality_matrix,inequality_vector, objective_vector,
                                                                   self.datacenters[datacenter].getBounds(demands, servers,
                                                                                                          actives))
        assert len(decision_variables)== len(servers)+num_active

        add = {}
        remove = {}

        for i in range(len(servers)):
            remove[ servers[i].name ] = decision_variables[i]

        activeCount = 0
        for i in range(len(actives)):
            if actives[i]:
                add[ servers[i].name] = decision_variables[len(servers)+activeCount]
                activeCount += 1

        return add, remove
    
    



    
    def calculate_normalized_lifespan(datacenter):
        total_servers = 0
        lifespan_sum = 0

        for server_type, server in datacenter.server_types.items():

            #retrieve the list of operating times for all servers of this type in the datacenter
            servers = datacenter.inventory[server_type]

            for deployed_time in servers:
                total_servers += 1
                
                #add the ratio of the server's operating time to its life expectancy to sum
                lifespan_sum += deployed_time / server.life_expectancy

        #calculate the normalized lifespan (L) as the average ratio across all servers
        return lifespan_sum / total_servers if total_servers > 0 else 0 # If there are no servers, return 0 to avoid division by zero
    

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
                #calculate met demand as the minimum of adjusted capacity and demand value
                met_demand = min(adjusted_capacity, demand_value)

                #calculate utilization ratio for this time step
                utilization_ratio = met_demand / adjusted_capacity

                #add utilization ratio to the total sum
                utilization_sum += utilization_ratio

                total_pairs += 1

        # calculate average utilization U
        return utilization_sum / total_pairs if total_pairs > 0 else 0 # If there are no pairs, return 0 to avoid division by zero.

    

    #  #get demand for each server and timestamp
    # servers = get_known('server_generation') 
    # for server in servers:
    #     high_demand = []
    #     medium_demand = []
    #     low_demand = []
    #     for ts in range(1, get_known('time_steps')+1):
    #         server_df = actual_demand.loc[(actual_demand['time_step']==ts) 
    #                                       & (actual_demand['server_generation']==server)].copy()
            
    #         #There is no demand of this particular server at the current timestamp
    #         if server_df.empty:
    #             high_demand.append(0)
    #             medium_demand.append(0)
    #             low_demand.append(0)
    #         else:
    #             high_demand.append(server_df.iloc[0]['high'])
    #             medium_demand.append(server_df.iloc[0]['medium'])
    #             low_demand.append(server_df.iloc[0]['low'])