from helpers.datacenters import Datacenter
from helpers.server_type import Server

class DecisionMaker(object):
    def __init__(self):
        self.datacenters = dict()
        self.server_types = dict()
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
