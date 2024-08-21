from helpers.datacenters import Datacenter
from helpers.server_type import Server

class DecisionMaker(object):
    def __init__(self):
        self.datacenters = dict()
        self.server_types = dict()
        self.id = 0
        self.timestep = 0

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

    def buyServer(self, datacenter: str, server_type: str) -> None:
        pass

    def sellServer(self, datacenter: str, server_type: str) -> None:
        pass

    def checkConstraints(self, datacenter: Datacenter) -> None:
        pass
