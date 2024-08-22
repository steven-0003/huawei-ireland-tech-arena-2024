from helpers.server_type import Server

class Datacenter(object):
    def __init__(self, name: str, cost_of_energy: float, latency_sensitivity: str, slots_capacity: int, 
                 server_types: dict[str,Server]) -> None:
        self.name = name
        self.cost_of_energy = cost_of_energy
        self.latency_sensitivity = latency_sensitivity
        self.slots_capacity = slots_capacity

        self.server_types = server_types
        self.inventory = {}
        self.inventory_level = 0

    def buy_server(self, server_type: str, id: str, timestep: int) -> None:
        assert(server_type in self.server_types.keys())
        
        if server_type not in self.inventory:
            self.inventory[server_type] = []

        self.inventory[server_type].append([timestep,id])
        self.inventory_level += self.server_types[server_type].slots_size

    def sell_server(self, server_type: str) -> str:
        assert(server_type in self.server_types.keys())
        assert(server_type in self.inventory.keys())

        self.inventory_level -= self.server_types[server_type].slots_size
        return self.inventory[server_type].pop(0)[1]
    
    def check_lifetime(self, cur_timestep: int) -> None:
        for server_type in self.inventory.keys():
            for i in range(len(self.inventory[server_type])):
                deployed_timestep = self.inventory[server_type][i][0]
                if cur_timestep - deployed_timestep > self.server_types[server_type].life_expectancy:
                    self.inventory[server_type].pop(i)
                else:
                    break