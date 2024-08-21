
class datacenter(object):
    def __init__(self, name, cost_of_energy, latency_sensitivity, slots_capacity):
        self.name = name
        self.cost_of_energy = cost_of_energy
        self.latency_sensitivity = latency_sensitivity
        self.slots_capacity = slots_capacity

        self.inventory = []
        self.inventory_level = 0

    def buy_server(self):
        pass

    def sell_server(self):
        pass

    def move_server(self):
        pass

    def add_to_inventory(self, server):
        pass