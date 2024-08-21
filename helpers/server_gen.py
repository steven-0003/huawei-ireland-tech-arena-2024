
class Server(object):
    def __init__(self, name, release_time, purchase_price, slots_size, energy_consumption, capacity, life_expectency,
                 cost_of_moving, avg_maintenance_fee, latency_sensitivity, selling_price):
        self.name = name
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

    # def score(self):
    #     time_active = self.deployed_timestep - self.release_time
      
    #     remaining_life = self.life_expectency - time_active
        
    #     if remaining_life <= 0:
    #         # If the server's life expectancy is exceeded, it should have the lowest possible score
    #         return float('-inf')

    #     profit_per_space = (self.selling_price - self.purchase_price) / self.slots_size
        
    #     #  How long the server will last ( `d` can be removed, but will keep it as a parameter)
    #     life_factor = d * remaining_life
        
    #     life_completion_ratio = remaining_life / self.life_expectency

    #     #  overall score 
    #     score = profit_per_space * life_factor * life_completion_ratio
        
    #     return score
