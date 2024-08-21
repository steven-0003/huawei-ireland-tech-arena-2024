
class decision_maker(object):
    def __init__(self):
        self.datacenters = {}
        self.server_types = {}
        self.id = 0

    def generateUniqueId(self):
        id = "server-" + self.id
        self.id += 1
        return id
    
    def addDataCenters(self, datacenters):
        for datacenter in datacenters:
            self.datacenters[datacenter.name] = datacenter

    def addServerTypes(self, server_types):
        for server_type in server_types:
            self.server_types[server_type.name] = server_type
