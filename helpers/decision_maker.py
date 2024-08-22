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
                "server_generation": server_type,
                "server_id": server_id,
                "action": action}
    
    def getLatencyDataCenters(self, latency_sensitivity: str) -> dict[str,Datacenter]:
        return {d: self.datacenters[d] for d in self.datacenters 
                if self.datacenters[d].latency_sensitivity == latency_sensitivity }
