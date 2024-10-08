import pulp
from evaluation import  get_known

from helpers.datacenters import Datacenter
from helpers.server_type import Server

class moveLP:

    def __init__(   self,
                    datacenters: dict[str, Datacenter],
                    server_types: dict[str, Server],
                    demand, timestep: int,
                    p,
                    predicted_demand: dict[str, Server] = None,
                    lifetimes_left:dict[str, dict[str,int]] = None,
                    ):
        
        self.datacenters = datacenters
        self.server_types = server_types
        self.demand  = demand
        self.predicted_demand = predicted_demand
        self.lifetimes_left = lifetimes_left
        self.p = p
            
        self.timestep = timestep

        ## create model 
        self.model = pulp.LpProblem("moves", pulp.LpMaximize)
        self.solver = pulp.PULP_CBC_CMD(msg=0)

        self.createVariables()
        self.setBounds()
        self.createConstraints()
        self.createObjective()
        self.createElasticDemandConstraints()
        
    def createVariables(self):

        ## create move decision variables 
        self.moveVariables = pulp.LpVariable.dicts(     name = "move",
                                                    indices = [
                                                        (dc_from.name+"_"+dc_to.name+"_"+server.name)
                                                            for dc_from in self.datacenters.values()
                                                                for dc_to in self.datacenters.values()
                                                                    for server in self.server_types.values()
                                                            if dc_from != dc_to and dc_from.latency_sensitivity!=dc_to.latency_sensitivity
                                                        ],
                                                    
                                                    cat = "Integer"
                                            )

        ## varaibles for how much of each server we will add to each datacenter
        self.addVariables = pulp.LpVariable.dicts( 
            name = "add",
            indices=[
                (dc.name+"_"+server.name) 
                for dc in self.datacenters.values()
                for server in self.server_types.values()
                if  server.canBeDeployed(self.timestep) and  server.isProfitable(self.timestep, dc.latency_sensitivity)

            ],

            cat="Integer"
        )

        ## varaibles for how much of each server we will remove from each datacenter
        self.removeVariables = pulp.LpVariable.dicts( 
            name = "remove",
            indices=[
                (dc.name+"_"+server.name) 
                for dc in self.datacenters.values()
                for server in self.server_types.values()
            ],

            cat="Integer"
        )

        ## variables for how much of each server we will hold at each datacenter
        self.holdVariables = pulp.LpVariable.dicts(
            name="hold",
            indices=[
                (dc.name+"_"+server.name) 
                for dc in self.datacenters.values()
                for server in self.server_types.values()
            ],
            cat="Integer"
        )

        ## variables to indicate whether we are increasing the number of servers of a particular type at a datacenter
        self.increaseVariables = pulp.LpVariable.dicts(
            name = "increase",
            indices= [
                (dc.name + "_"+ server.name) 
                for dc in self.datacenters.values()
                for server in self.server_types.values()
            ],
            lowBound=0,
            upBound=1,
            cat = "Binary"
        )

        ## variables to indicate whether we are decreasing the number of servers of a particular type at a datacenter
        self.decreaseVariables = pulp.LpVariable.dicts(
            name = "decrease",
            indices= [
                (dc.name + "_"+ server.name) 
                for dc in self.datacenters.values()
                for server in self.server_types.values()
            ],
            lowBound=0,
            upBound=1,
            cat = "Binary"
        )
        
    def setBounds(self):

        ## set bounds for variables

        ## move variables bounds
        for var in self.moveVariables:

            self.moveVariables[var].lowBound = 0
            var_details = var.split("_")

            self.moveVariables[var].upBound =   self.datacenters[ var_details[0] ].getServerStock( var.split("_")[2] ) 

        ## add variable bounds
        for var in self.addVariables:
            self.addVariables[var].lowBound=0
            var_details = var.split("_")
            
            self.addVariables[var].upBound= int(self.datacenters[var_details[0]].slots_capacity
                                                /self.server_types[var_details[1]].capacity)

        ## remove variable bounds
        for var in self.removeVariables:    
            self.removeVariables[var].lowBound=0
            var_details = var.split("_")
            self.removeVariables[var].upBound = self.datacenters[var_details[0]].getServerStock(var_details[1])

        ## hold variable bounds
        for var in self.holdVariables:
            self.holdVariables[var].lowBound=0
            var_details = var.split("_")
            self.holdVariables[var].upBound = self.datacenters[var_details[0]].getServerStock(var_details[1])

    def createConstraints(self):

        self.createIncreaseVariableConstraints()

        self.setAddIncreaseBounds()
        self.setRemoveIncreaseBounds()
        self.setMoveIncreaseBounds()
        
        ## the sum of all the servers moving from f should be less than the number of servers at f  
        for s in self.server_types:
            for f,dc in self.datacenters.items():

                self.model += (
                                    pulp.lpSum( [ self.moveVariables[var] for var in self.moveVariables if var.split("_")[0]==f  and var.split("_")[2]==s] 
                                            +   [self.removeVariables[var] for var in self.removeVariables if var.split("_")[0]==f and var.split("_")[1]==s]
                                        )
                            
                                            <= dc.getServerStock(s),
                                        
                                        f+s+" Move + Add Less Than Current Stock Constraint"
                            )
                
        ## the sum of all the servers moving from f, the servers being removed from f and the servers being held at f
        ## should be equal to the current stock of that server
        for s in self.server_types:
            for f,dc in self.datacenters.items():
                
                self.model += (pulp.lpSum(  [ self.moveVariables[var] for var in self.moveVariables if var.split("_")[0]==f  and var.split("_")[2]==s] 
                                          + [self.removeVariables[var] for var in self.removeVariables if var.split("_")[0]==f and var.split("_")[1]==s] 
                                          + [self.holdVariables[var] for var in self.holdVariables if var.split("_")[0]==f and var.split("_")[1] == s]
                                          ) == dc.getServerStock(s),
                                f+s+" -Move - Remove + Hold Equal To Current Stock Constraint")
                      
        ## the number of servers of a particular type moving to a datacentre, shouldnt exceed the demand*CONSTANT for that server at that datacenter
        for s in self.server_types:            
            for latency in get_known('latency_sensitivity'):

                self.model +=         (         
                                            pulp.lpSum(  
                                                [   self.moveVariables[var] * self.server_types[var.split("_")[2]].capacity 
                                                    for var in self.moveVariables
                                                    if self.datacenters[var.split("_")[1]].latency_sensitivity==latency and var.split("_")[2]==s ]

                                            +   [   self.addVariables[var] * self.server_types[var.split("_")[1]].capacity
                                                    for var in self.addVariables
                                                    if self.datacenters[var.split("_")[0]].latency_sensitivity==latency and var.split("_")[1]==s ]

                                           
                                            +
                                                [   self.holdVariables[var] * self.server_types[var.split("_")[1]].capacity
                                                    for var in self.holdVariables
                                                    if self.datacenters[var.split("_")[0]].latency_sensitivity == latency and var.split("_")[1]==s]
                                            ) 
                                    <=
                                    self.demand[latency][s] * 2
                                 
                                , s+latency+ " Adds - Removes + Moves to datacenters of this latency shouldnt exceed the demand Constraint"
                                )
                
        ## the total slot size of the servers going in and out of a datacenter should not exceed the slot capacity 
        for t, dc in self.datacenters.items():
                
                self.model += (
                                    pulp.lpSum(
                                        [  
                                              (self.moveVariables[var] * self.server_types[var.split("_")[2]].slots_size )
                                                for var in self.moveVariables
                                                if var.split("_")[1]==t
                                        ]           
                                        +
                                        [
                                            (self.addVariables[var] * self.server_types[var.split("_")[1]].slots_size)
                                                for var in self.addVariables
                                                if var.split("_")[0]==t
                                        ]                                 
                                        +
                                        [
                                           (self.holdVariables[var] * self.server_types[var.split("_")[1]].slots_size )
                                                for var in self.holdVariables
                                                if var.split("_")[0] == t
                                        ]
                                    )

                                    <= dc.slots_capacity

                                    , t + " Slots Taken Do Not Exceed Capacity Constraint"
                            )

    def createIncreaseVariableConstraints(self):

        ## initialises increase and decrease variables
        for var in self.increaseVariables:
            self.model += self.increaseVariables[var]>=0
        for var in self.decreaseVariables:
            self.model += self.decreaseVariables[var]>=0

        ## makes sure increase and decrease variables cannot be the same as one another
        for dc in self.datacenters:
            for s in self.server_types:
                self.model += self.increaseVariables[dc+"_"+s] + self.decreaseVariables[dc+"_"+s] == 1 
    
    def setAddIncreaseBounds(self):

        ## ensures that add variables are only larger than 0 if the corresponding increase variable is 1
        for var in self.addVariables:
            
            var_details = var.split("_")
            latency = self.datacenters[var_details[0]].latency_sensitivity

            self.model += self.addVariables[var] <= self.demand[latency][var_details[1]] * self.increaseVariables[var_details[0]+"_"+var_details[1]]

    def setRemoveIncreaseBounds(self):

         ## ensures that remove variables are only larger than 0 if the corresponding increase variable is 1
        for var in self.removeVariables:
            
            var_details = var.split("_")
            self.model += self.removeVariables[var] <= self.datacenters[var_details[0]].getServerStock(var_details[1]) *( self.decreaseVariables[var_details[0]+"_"+var_details[1]])
        
    def setMoveIncreaseBounds(self):

        ## makes sure that move variables to and from are set to 0 according to whether or not that datacenter is increasing for a particular server
        for var in self.moveVariables:

            var_details = var.split("_")
            self.model += self.moveVariables[var] <=   self.datacenters[ var_details[0] ].getServerStock( var.split("_")[2] ) * (self.decreaseVariables[var_details[0]+"_"+var_details[2]])

        for var in self.moveVariables:
            
            var_details = var.split("_")
            latency = self.datacenters[var_details[1]].latency_sensitivity
            self.model += self.moveVariables[var] <= self.demand[latency][var_details[2]] * self.increaseVariables[var_details[1]+"_"+var_details[2]]

    def createElasticDemandConstraints(self):
        
        ## the number of servers of a particular type moving to a datacentre, shouldnt exceed the demand for that server at that datacenter
        for s in self.server_types:            
            for latency in get_known('latency_sensitivity'):
        
                constraint= pulp.LpConstraint           (
                                e=pulp.lpSum(  
                                                [   self.moveVariables[var] * self.server_types[var.split("_")[2]].capacity 
                                                    for var in self.moveVariables
                                                    if self.datacenters[var.split("_")[1]].latency_sensitivity==latency and var.split("_")[2]==s ]
                                            +
                                                [   self.addVariables[var] * self.server_types[var.split("_")[1]].capacity
                                                    for var in self.addVariables
                                                    if self.datacenters[var.split("_")[0]].latency_sensitivity==latency and var.split("_")[1]==s ]
                                            +
                                                [   self.holdVariables[var] * self.server_types[var.split("_")[1]].capacity
                                                    for var in self.holdVariables
                                                    if self.datacenters[var.split("_")[0]].latency_sensitivity == latency and var.split("_")[1]==s]
                                            ) 
                      
                                    , sense=pulp.LpConstraintLE
                                    
                                , name=s+latency+ " Adds - Removes + Moves to datacenters of this latency shouldnt exceed the demand Constraint"
                                ,rhs=self.demand[latency][s]
                                )
                
                datacenters = [dc for dc in self.datacenters.values() if dc.latency_sensitivity==latency]

                energy_costs = [dc.cost_of_energy for dc in datacenters]

                avg_cost_of_energy = sum(energy_costs)/len(energy_costs)

                server = self.server_types[s]
                profit = -server.selling_prices[latency] * server.capacity - ( (server.energy_consumption * avg_cost_of_energy) )

                lifetimes_left = [  self.lifetimes_left[dc.name][s] for dc in datacenters]
                avg_lifetime_left = sum(lifetimes_left)/len(datacenters)

                lifetime_coeff = 1- (avg_lifetime_left/server.life_expectancy)

                profit *= lifetime_coeff

                profit *= self.p

                profit *= ((self.demand[latency][s]+1)/(self.predicted_demand[latency][s]+1))

                econstraint = constraint.makeElasticSubProblem(profit,proportionFreeBound = 0)
                self.model.extend(econstraint)
                
    def createObjective(self):
        
        self.model += pulp.lpSum(
            [
                self.getAddObjectiveCoeff(var) * self.addVariables[var] for var in self.addVariables
            ]
            +
            [
                self.getMoveObjectiveCoeff(var) * self.moveVariables[var] for var in self.moveVariables
            ]
            +
            [
                self.getRemoveObjectiveCoeff(var) * self.removeVariables[var] for var in self.removeVariables
            ]
            +
            [
                self.getHoldObjectiveCoeff(var) * self.holdVariables[var] for var in self.holdVariables
            ]
        )

    def getAddObjectiveCoeff(self, var):
        var_details = var.split("_")
        
        datacenter = self.datacenters[var_details[0]]
        latency = datacenter.latency_sensitivity
        server = self.server_types[var_details[1]]
       
        profit = server.selling_prices[latency] * server.capacity - ( (server.energy_consumption * datacenter.cost_of_energy))
        profit = int(profit / server.life_expectancy)+1

        pred_demand =   self.predicted_demand[latency][server.name]                                  
        demand =   self.demand[latency][server.name]

        demand_term =  (pred_demand-demand+1)/(pred_demand+demand+1)

        profit *= demand_term

        return profit 
      
    def getMoveObjectiveCoeff(self, var):
        var_details = var.split("_")

        server = self.server_types[var_details[2]]

        from_dc = self.datacenters[var_details[0]]
        to_dc = self.datacenters[var_details[1]]
        
        from_latency_dcs = [dc for dc in self.datacenters.values() if dc.latency_sensitivity == from_dc.latency_sensitivity ]
        from_demand_met =  sum([ dc.getServerStock(server.name)*server.capacity  for dc in from_latency_dcs])

        to_latency_dcs = [dc for dc in self.datacenters.values() if dc.latency_sensitivity==to_dc.latency_sensitivity ]
        to_demand_met = sum( [dc.getServerStock(server.name)*server.capacity for dc in to_latency_dcs ] )

        ## account for demand of the latency we are transferring to 
        ## as well as the demand of the latency we are moving from 
        from_total_demand = self.demand[from_dc.latency_sensitivity][server.name]
        to_total_demand = self.demand[to_dc.latency_sensitivity][server.name]

        from_demand_coeff = (from_demand_met-from_total_demand+1)/(from_demand_met+from_total_demand+1)
        to_demand_coeff = (to_total_demand-to_demand_met+1)/(to_total_demand+to_demand_met+1)

        demand_coeff = min(from_demand_coeff,to_demand_coeff)
        
        from_profit = (server.selling_prices[from_dc.latency_sensitivity]*server.capacity)- (server.energy_consumption*from_dc.cost_of_energy)
        to_profit = (server.selling_prices[to_dc.latency_sensitivity]*server.capacity) - (server.energy_consumption*to_dc.cost_of_energy)

        ## percentage profit increase
        profit_increase = (to_profit+1)/(from_profit+1)

        profit = to_profit * profit_increase * demand_coeff 

        ## add score increase from add obj func, to make sure move is always more favourable than add
        profit += (server.selling_prices[to_dc.latency_sensitivity] * server.capacity - ( (server.energy_consumption * to_dc.cost_of_energy)))

        return profit 

    def getRemoveObjectiveCoeff(self, var):
        var_details = var.split("_")
        
        datacenter = self.datacenters[var_details[0]]
        latency = datacenter.latency_sensitivity
        server = self.server_types[var_details[1]]
    
        profit = -(server.selling_prices[latency] * server.capacity - ( (server.energy_consumption * datacenter.cost_of_energy) ))

        lifetime_left = self.lifetimes_left[datacenter.name][server.name]
        lifetime_coeff = lifetime_left/server.life_expectancy

        profit *= lifetime_coeff
                  
        return profit

    def getHoldObjectiveCoeff(self, var):
        var_details = var.split("_")
        
        datacenter = self.datacenters[var_details[0]]
        latency = datacenter.latency_sensitivity
        server = self.server_types[var_details[1]]

        lifetime_left = self.lifetimes_left[datacenter.name][server.name]
        lifetime_coeff = lifetime_left/server.life_expectancy

        profit = server.selling_prices[latency] * server.capacity -  (server.energy_consumption * datacenter.cost_of_energy * (1-lifetime_coeff))
                                                                                                                                         
        return profit
       
    def solve(self):
        self.model.solve(self.solver)
    