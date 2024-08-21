
class Server(object):
    def __init__(self, id, name, release_time, purchase_price, slots_size, energy_consumption, capacity, life_expectency,
                 cost_of_moving, avg_maintenance_fee, latency_sensitivity, selling_price):
        self.name = name
        self.id = id
        self.name = name
        self.release_time = release_time
        self.purchase_price = purchase_price
        self.slots_size = slots_size
        self.energy_consumption = energy_consumption
        self.capacity = capacity
        self.life_expectency = life_expectency
        self.cost_of_moving = cost_of_moving
        self.avg_maintenance_fee = avg_maintenance_fee
        self.latency_sensitivity = latency_sensitivity
        self.selling_price = selling_price

    def score(self):
        return 0