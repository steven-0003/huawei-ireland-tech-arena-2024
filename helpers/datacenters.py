
from typing_extensions import Self
from helpers.server_type import Server

class Datacenter(object):
    """
    A class to represent a datacenter.


    Attributes
    ----------
    name : str
        The datacenter id
    cost_of_energy : float
        How much energy costs at this datacenter
    latency_sensitivity : str
        The latency sensitivity of this datacenter
    slots_capacity : int
        The number of slots that the datacenter can hold
    server_types : dict[str,Server]
        A dictionary of all server generations
    inventory : dict[str,list[list]]
        A dictionary of the stored inventory for each server generation
    inventory_level : int
        The number of slots that are currently being used

    """
    def __init__(self, name: str, cost_of_energy: float, latency_sensitivity: str, slots_capacity: int, 
                 server_types: dict[str,Server]) -> None:
        self.name = name
        self.cost_of_energy = cost_of_energy
        self.latency_sensitivity = latency_sensitivity
        self.slots_capacity = slots_capacity

        self.server_types = server_types

        self.inventory = {s : [] for s in server_types.keys()}
        self.inventory_level = 0

        self.adding_servers = {}
        self.removing_servers = {}

    def buy_server(self,  server_type: str, id:str , timestep: int) -> None:
        """Adds a server to the datacenter

        Args:
            server_type (str): The server generation that will be added
            id (str): The id of the server to be added
            timestep (int): The timestep that the server is added at
        """
        assert(server_type in self.server_types.keys())
  
        if server_type not in self.inventory:
            self.inventory[server_type] = []

        self.inventory[server_type].append((timestep,id))

        self.inventory_level += self.server_types[server_type].slots_size

        assert self.inventory_level <= self.slots_capacity

    def sell_server(self, server_type: str) -> str:
        """Removes a server from the datacenter

        Args:
            server_type (str): The server generation to be removed

        Returns:
            str: The id of the server that was removed
        """
        assert(server_type in self.server_types.keys())
        assert(server_type in self.inventory.keys())

        self.inventory_level -= self.server_types[server_type].slots_size

        assert self.inventory[server_type]

        id = self.inventory[server_type].pop(0)[1]

        return id
    
    def move_server(self, seller: Self, server_type: str) -> str:
        """Moves a server from another datacenter to this datacenter

        Args:
            seller (Self): The datacenter that we are moving a server from
            server_type (str): The server generation that is being moved

        Returns:
            str: The id of the server that was moved
        """
        assert(len(seller.inventory[server_type]) > 0)

        server = seller.inventory[server_type].pop()
        seller.inventory_level -= self.server_types[server_type].slots_size

        self.inventory[server_type].append(server)
        self.inventory[server_type].sort(key=lambda x: x[0])
        self.inventory_level += self.server_types[server_type].slots_size

        return server[1]

    def check_lifetime(self, cur_timestep: int) -> None:
        """Evict all servers that has exceeded its lifetime

        Args:
            cur_timestep (int): The current timestep that we are at
        """
        for server_type in self.inventory.keys():
            queue = self.inventory[server_type]

            while(len(queue) != 0 and cur_timestep - queue[0][0] >= self.server_types[server_type].life_expectancy):
                self.inventory[server_type].pop(0)
                self.inventory_level -= self.server_types[server_type].slots_size

    def remainingCapacity(self) -> int:
        """Gets the remaining capacity of this datacenter

        Returns:
            int: The number of slots that are free
        """
        return self.slots_capacity - self.inventory_level
            
    def getServerStock(self, server_name: str) -> int:
        """Gets the current server stock

        Args:
            server_name (str): The server generation

        Returns:
            int: The number of servers in the datacenter of a server generation
        """        
        return len(self.inventory.get(server_name, []))
    
    # 
    def getStockLevel(self) -> int:
        """Gets the current stock level of all servers

        Returns:
            int: The number of all servers in the datacenter
        """        

        sum = 0 

        for i in self.server_types:
            sum += self.getServerStock(i)

        return sum

    def getProfitability(self) -> float:
        """Used to rank the profitability of datacenters

        Returns:
            float: How profitable the datacenter is
        """
        if self.latency_sensitivity == 'low':
            return 1 * self.cost_of_energy
        if self.latency_sensitivity == 'medium':
            return 2 * self.cost_of_energy
        return 3 * self.cost_of_energy
