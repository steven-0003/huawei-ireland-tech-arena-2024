
class Server(object):
    def __init__(self, name: str, release_time: list[int], purchase_price: float, slots_size: int, 
                 energy_consumption: float, capacity: int, life_expectancy: int, cost_of_moving: float, 
                 avg_maintenance_fee: float) -> None:
        self.name = name
        self.release_time = release_time
        self.purchase_price = purchase_price
        self.slots_size = slots_size
        self.energy_consumption = energy_consumption
        self.capacity = capacity
        self.life_expectancy = life_expectancy
        self.cost_of_moving = cost_of_moving
        self.avg_maintenance_fee = avg_maintenance_fee

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
