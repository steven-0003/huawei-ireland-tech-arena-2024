from helpers.datacenters import Datacenter
from helpers.server_type import Server

import ast


class DecisionMaker(object):
    def __init__(self, datacenters, server_types, selling_prices):

        self.datacenters = dict()
        self.server_types = dict()

        ## create all server types, a server type is a cpu generation with a specific latency sensitivity 
        self.server_types = {s.server_generation+"_"+latency_sensitivity: Server(s.server_generation,ast.literal_eval(s.release_time),s.purchase_price, 
                                          s.slots_size, s.energy_consumption,s.capacity,s.life_expectancy,
                                          s.cost_of_moving,s.average_maintenance_fee, latency_sensitivity) for s in server_types.itertuples() for latency_sensitivity in ["low","medium","high"]}

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
        return {"time_step": timestep,
                "datacenter_id": datacenter,
                "server_generation": server_type.split('_')[0],
                "server_id": server_id,
                "action": action}
    
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
        return 1/2 * ((energy_cost/energy_cost_sum) + (remaining_capacity/remaining_capacity_sum))
    
    def getActiveServers(self) -> None:
        self.active_server_types = [server for server in self.server_types.keys() if self.server_types[server].canBeDeployed(self.timestep)]

    # Helper function to extract relevant server data
    def extractRelevantData(self, datacenter_name: str, server_demands: dict[str, int]) -> tuple:
        # check if datacenter exists
        if datacenter_name not in self.datacenters:
            raise ValueError(f"Datacenter '{datacenter_name}' does not exist.")
        
        # get datacenter
        datacenter = self.datacenters[datacenter_name]
        
        # initialize lists to store server data
        server_sizes = []
        server_stock = []

        # extract server data
        for server_type_name, server_type in self.server_types.items():
            server_sizes.append(server_type.slots_size)
            server_stock.append(len(datacenter.inventory.get(server_type_name, [])))

        # make sure demands are in the same order as the server types (dont know if we need this)
        demands_list = [server_demands[server_type_name] for server_type_name in self.server_types.keys()]

        return server_sizes, demands_list, server_stock
    

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