from typing import Dict

from evaluation import get_known

class Server(object):
    """
    A class to represent a server generation.

    Attributes
    ----------
    name : str
        The server generation
    release_time : list[int]
        The timeframe that the server can be released in
    purchase_price : float
        The cost to purchase the server
    slots_size : int
        The number of slots that the server uses
    energy_consumption : float
        The amount of energy the server consumes
    capacity : int
        The capacity of the server
    life_expectancy : int
        How long the server is expected to live for in timesteps
    cost_of_moving: float
        How much it costs to move the server to another datacenter
    avg_maintenance_fee : float
        The average maintenance fee of the server
    k: int
        A parameter that is used in getTimeTillProfitable
    selling_prices : Dict
        A dictionary of selling prices for each latency sensitivity

    """
    def __init__(self, name: str, release_time: list[int], purchase_price: float, slots_size: int, 
                 energy_consumption: float, capacity: int, life_expectancy: int, cost_of_moving: float, 
                 avg_maintenance_fee: float,  k: int, selling_prices:Dict = {}) -> None:
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

    def setSellingPrices(self, new_prices: Dict) -> None:
        self.selling_prices = new_prices

    def canBeDeployed(self, cur_timestep: int) -> bool:
        """Is the server generation within its release time frame

        Args:
            cur_timestep (int): The current timestep

        Returns:
            bool: True if the server is within its release time frame
        """        
        return(cur_timestep >= self.release_time[0] and cur_timestep <= self.release_time[1])

    def isProfitable(self, timestep:int, latency: str) -> bool:
        """Whether the server is profitable or not

        Args:
            timestep (int): The current timestep
            latency (str): The latency sensitivity that requires this server generation

        Returns:
            bool: True if the server is profitable
        """        
        
        return get_known("time_steps")- self.getTimeTillProfitable(latency_sensitivity=latency) > timestep           

    def getTimeTillProfitable(self, latency_sensitivity: str) -> float:
        """The time until the server is profitable

        Args:
            latency_sensitivity (str): The latency sensitivity that requires this server generation

        Returns:
            float: The amount of time the server should be deployed for
        """       

        return  (self.purchase_price /  (self.selling_prices[latency_sensitivity] * self.capacity)) + self.k
