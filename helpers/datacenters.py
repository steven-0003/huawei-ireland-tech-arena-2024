
class datacenter(object):
    def __init__(self, name, cost_of_energy, latency_sensitivity, slots_capacity, server_types):
        self.name = name
        self.cost_of_energy = cost_of_energy
        self.latency_sensitivity = latency_sensitivity
        self.slots_capacity = slots_capacity

        self.server_types = server_types
        self.inventory = {}
        self.inventory_level = 0

    def buy_server(self, server_type, id, timestep):
        #invalid server type
        if server_type not in self.server_types.keys():
            return False

        #not enough space
        if self.inventory_level + self.server_types[server_type].slots_size >= self.slots_capacity:
            return False
        
        self.inventory[server_type].append([id,timestep])
        self.inventory_level += self.server_types[server_type].slots_size
        return True

    def sell_server(self, server_type):
        #invalid server type
        if server_type not in self.server_types.keys():
            return False

        #nothing to sell
        if self.inventory_level == 0:
            return False

        self.inventory[server_type].pop(0)
        self.inventory_level -= self.server_types[server_type].slots_size
        return True
    
    def check_lifetime(self, cur_timestep):
        for server_type in self.inventory.keys():
            for i in range(len(self.inventory[server_type])):
                deployed_timestep = self.inventory[server_type][i][0]
                if cur_timestep - deployed_timestep > self.server_types[server_type].lifetime:
                    self.inventory[server_type].pop(i)
                else:
                    break