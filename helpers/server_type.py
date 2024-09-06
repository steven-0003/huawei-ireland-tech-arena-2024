
from typing import Dict

from evaluation import get_known


class Server(object):
    def __init__(self, name: str, release_time: list[int], purchase_price: float, slots_size: int, 
                 energy_consumption: float, capacity: int, life_expectancy: int, cost_of_moving: float, 
                 avg_maintenance_fee: float,  k, selling_prices:Dict = {}) -> None:
        self.name = name
        self.release_time = release_time
        self.purchase_price = purchase_price
        self.selling_prices = selling_prices
        self.slots_size = slots_size
        self.energy_consumption = energy_consumption
        self.capacity = capacity
        self.life_expectancy = life_expectancy
        self.cost_of_moving = cost_of_moving
        self.avg_maintenance_fee = avg_maintenance_fee
        self.k = k
        


    def setSellingPrices(self, new_prices):
        self.selling_prices = new_prices

    def canBeDeployed(self, cur_timestep):
        return(cur_timestep >= self.release_time[0] and cur_timestep <= self.release_time[1])


    def isProfitable(self, timestep, latency):
        
        return get_known("time_steps")- self.getTimeTillProfitable(latency_sensitivity=latency) > timestep
            
        

    def getTimeTillProfitable(self, latency_sensitivity):

        k = self.k

        return  (self.purchase_price /  (self.selling_prices[latency_sensitivity] * self.capacity)) + k