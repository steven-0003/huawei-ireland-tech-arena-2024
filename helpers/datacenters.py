from typing import List
from helpers.server_type import Server
import linear_programming


 

class Datacenter(object):
    def __init__(self, name: str, cost_of_energy: float, latency_sensitivity: str, slots_capacity: int, 
                 server_types: dict[str,Server]) -> None:
        self.name = name
        self.cost_of_energy = cost_of_energy
        self.latency_sensitivity = latency_sensitivity
        self.slots_capacity = slots_capacity

        self.server_types = server_types
        ## do we need a dict for server types, could this be a list 
        # self.server_types.sort(key= lambda x: x.name ) ## sort alphabetically to ensure consistent order

        self.inventory = {s : [] for s in server_types.keys()}
        self.inventory_level = 0

        self.adding_servers = {}
        self.removing_servers = {}

    def buy_server(self,  server_type: str, id:str , timestep: int) -> None:
        assert(server_type in self.server_types.keys())

        
        if server_type not in self.inventory:
            self.inventory[server_type] = []

        
        self.inventory[server_type].append((timestep,id))

        self.inventory_level += self.server_types[server_type].slots_size

        assert self.inventory_level <= self.slots_capacity






    def sell_server(self, server_type: str) -> str:
        assert(server_type in self.server_types.keys())
        assert(server_type in self.inventory.keys())

        self.inventory_level -= self.server_types[server_type].slots_size

        assert self.inventory[server_type]

        id = self.inventory[server_type].pop(0)[1]

        return id
    
  
    
    def check_lifetime(self, cur_timestep: int) -> None:
        for server_type in self.inventory.keys():

            for i in range( len(self.inventory[server_type]) ):

                ## get timestep at which last item in inventory for this server was dedployed
                deployed_timestep = self.inventory[server_type][i][0]
                
                ## if server has exceeded its lifetime remove it 
                if cur_timestep - deployed_timestep > self.server_types[server_type].life_expectancy:
                    self.inventory[server_type].pop(i)
                    self.inventory_level -= self.server_types[server_type].slots_size
                else:

                    break

    def remainingCapacity(self) -> int:
        return self.slots_capacity - self.inventory_level
    
    def getBounds(self, demand: List[int], servers: List[Server], actives: List[bool]) -> List[tuple]:

        ## could this function potentially remove the servers parameter, and use the server_types in this class
        ## if this is done we just have to watch out that this uses the same ordering as used in getAddRemove()

        bounds = []
        for i in range(len(servers)):
            bounds.append((0,len(self.inventory.get(servers[i].name, []))))
        
        activeCount = 0
        for i in range(len(servers)):
            if actives[i]:
                bounds.append((0,demand[i]//servers[i].capacity))

        return bounds
    


    ## timestep = the current timestep
    ## this assumes the servers are sorted alphabetically
    ## demands = a dictionary of server types to demand 
    def find_add_remove_for_all_servers(self, timestep , demands):
        
        for server in self.server_types: 
            self.adding_servers, self.removing_servers = self.getAddRemove(demands, timestep)




    def getAddRemove(self, demands: list[int], timestep):


        ## get and sort servers 
        servers = list(self.server_types.values())
        
        servers.sort(key = lambda x: x.name )

        ## get list of bools indicating whether each server is active or not
        actives = [server.canBeDeployed(timestep) for server in servers]

        num_active = len([a for a in actives if a ==True ])

        ## gets the list of demands for each server, in the correct order
        demands = [ demands[s.name] for s in servers]

        remaining_slots = self.remainingCapacity()
        current_server_stock = [ len(self.inventory.get(server.name, [])) for server in servers]

        server_capacities = [ s.capacity for s in servers ]

        ## create matrices for linear programming solution
        inequality_matrix = linear_programming.create_inequality_matrix(servers, actives )

        inequality_vector = linear_programming.create_inequality_vector( 
                                                                            remaining_slots,
                                                                            demands,
                                                                            current_server_stock,
                                                                            server_capacities
                                                                        )
        
        objective_vector = linear_programming.create_objective_vector(  servers,
                                                                        actives,
                                                                        self.cost_of_energy,
                                                                        self.latency_sensitivity
                                                                    )
        ## find "optimal" decision variables
        decision_variables = linear_programming.find_add_and_evict( inequality_matrix,
                                                                    inequality_vector,
                                                                    objective_vector,
                                                                    self.getBounds(demands, servers,actives)
                                                                )
        

        assert len(decision_variables)== len(servers)+num_active

        add = {}
        remove = {}

        for i in range(len(servers)):
            remove[ servers[i].name ] = decision_variables[i]

        activeCount = 0
        for i in range(len(actives)):
            if actives[i]:
                add[ servers[i].name] = decision_variables[len(servers)+activeCount]
                activeCount += 1

        return add, remove


            

        







