import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os

class IndividualFreerider:
    def __init__(self, node_id, individual_type='Cooperative'):
        self.node_id = node_id
        self.type = individual_type  #Freerider or Cooperative

class IndividualKnow:
    def __init__(self, node_id, individual_type='DontKnow'):
        self.node_id = node_id
        self.type = individual_type  #Know or Don't Know


class Resources:
    def __init__(self, initial_value, replenishment_proportion, coop_consumption_rate, freerider_slope, freerider_intercept):
        """
        Parameters:
        - initial_value: The starting amount of the resource.
        - replenishment_amount: Amount of resources (per capita) to replenish at each step.
        - consumption_rates: Dictionary defining how much each type of individual consumes.
        - coop_consumption_rate: Rate of resource consumption by cooperatives (NOTE this rate incorporates the proportion that are cooperative - because constant)
        - freerider_slope: slope of freerider consumption as a function of proportion of individuals that know
        - freerider_intercept: intercept of freerider consumption as a function of proportion of individuals that know
        """
        self.value = initial_value
        self.replenishment_amount = replenishment_proportion
        self.coop_consumption_rate = coop_consumption_rate
        self.freerider_slope = freerider_slope
        self.freerider_intercept = freerider_intercept

    def update(self, proportions):
        """
        Update the resource value based on consumption and replenishment.

        Parameters:
        - proportions: A dictionary with the proportion of individuals in each type (Know, DontKnow).
        """
        total_consumption = 0

        # TODO CHECK this function - Calculate total consumption based on freerider consumption function  (of the proportion of population that know) and the constant cooperative consumption rate TODO
            total_consumption = (self.freerider_slope * proportions.Know + self.freerider_intercept ) + self.coop_consumption_rate

        # Replenishment is a proportion of the current resource level
        self.value = self.value - total_consumption + self.replenishment_amount

        # Ensure resource value does not go negative
        if self.value < 0:
            self.value = 0


    def get_value(self):
        return self.value


class ABMModel:
    def __init__(self, graph, p_freeriders, beta, mu, T, initial_resource, replenishment_amount, consumption_rates):
        """
        Initializes the model with a given graph and other parameters.

        Parameters:
        - graph: The networkx graph to be used in the simulation.
        - p_freeriders: Fraction of individuals that are freeriders (constant across all simulations)
        - beta: Probability parameter for knowing about freeriders (Know).
        - mu: Constant probability of forgetting about freeriders.
        - T: Number of time steps to simulate.
        - initial_resource: Initial value of the resource pool (per capita).
        - replenishment_amount: Amount of resources (per capita) to replenish at each step.
        - consumption_rates: Dictionary of consumption rates for each individual type.
        """
        self.graph = graph
        self.num_individuals = graph.number_of_nodes()
        self.p_freeriders = p_freeriders
        self.beta = beta
        self.mu = mu  # Replaces gamma with mu
        self.T = T
        self.freeriderindividuals = {i: Individual(i) for i in self.graph.nodes()}
        self.knowindividuals = {i: Individual(i) for i in self.graph.nodes()}
        self.initialize_freeriderindividuals()
        self.initialize_knowindividuals()
        self.know_history = []  # Store the proportions at each time step
        self.resource_history = []  # Store resource values at each step

        # Initialize the Resources class
        self.resources = Resources(initial_resource, replenishment_amount, coop_consumption_rate, freerider_slope, freerider_intercept)

    def initialize_freeriderindividuals(self):
        """Initialize individuals, all are Cooperative and a fraction p are freeriders"""
        for i in self.freeriderindividuals:
            self.freeriderindividuals[i].type = 'Cooperative'

        # Set initial freerider individuals based on fraction p_freerider
        initial_freeriders = random.sample(list(self.freeriderindividuals.keys()), int(self.p_freeriders * self.num_individuals))
        for i in initial_freeriders:
            self.freeriderindividuals[i].type = 'Freeriders'
    
    def initialize_knowindividuals(self):
        """Initialize the individuals that know about freeriders - the direct neighbours of the freeriders. The rest Dont Know. TODO This should exclude the freeriders """
        for i in self.knowindividuals:
            self.knowindividuals[i].type = 'DontKnow'

        # Set initial know individuals that are neighbours of freeriders
        freeriders = list() #TODO this needs to list all nodes that are freeriders (to find their neighbors)
        neighbours = set()
        for node in freeriders:
            neighbours.update(graph.neighbors(node))
            initial_know = list(neighbors - set(freeriders)) #TODO I'm not sure this is right - maybe I need to return initial_know

        for i in initial_know:
            self.knowindividuals[i].type = 'Know'

    def step(self, t):
        """Perform a single step of the simulation."""
        #TODO we need to exclude the freeriders from this part where individuals transition between know and don't know
        for i in self.knowindividuals:
            random_value = np.random.random()
            neighbors = list(self.graph.neighbors(i))
            num_know_neighbors = sum([1 for n in neighbors if self.knowindividuals[n].type == 'Know'])

            # Probability to become Know
            prob_know = 1 - (1 - self.beta) ** num_know_neighbors

            # Apply state changes based on probabilities TODO we need to exclude the freeriders from this
            if self.knowindividuals[i].type == 'DontKnow':
                if random_value < prob_know:
                    self.knowindividuals[i].type = 'Know'
            elif self.knowindividuals[i].type == 'Know':
                if random_value < self.mu:  # Use mu instead of gamma
                    self.knowindividuals[i].type = 'DontKnow'

    def run(self):
        """Run the simulation for T steps."""
        for t in range(self.T):
            self.step(t)
            self.record_state()

    def record_state(self):
        """Record the proportion of each type of individual and the resource value."""
        state_counts = {'DontKnow': 0, 'Know': 0}
        for individual in self.knowindividuals.values():
            state_counts[individual.type] += 1

        # Normalize to get proportions (TODO we need to minus the number of freeriders from numerator and denominator)
        proportions = {key: state_counts[key] / self.num_individuals for key in state_counts}
        self.history.append(proportions)

        # Update the resource pool based on the proportion of individuals
        self.resources.update(proportions)
        self.resource_history.append(self.resources.get_value())

    def get_history(self):
        """Return the history of proportions at each time step."""
        return self.history

    def get_resource_history(self):
        """Return the history of resource values."""
        return self.resource_history




# TODO: automatize the scaling process for resources dynamics
# Simulation Parameters
num_individuals = 100  # Number of individuals in the graph
p_freeriders = 0.05  # Initial fraction of cooperative individuals
beta = 0.8  # Probability parameter for cooperation
mu = 0.2  # Constant probability of becoming indifferent
T = 200  # Number of time steps

# Resource parameters
initial_resource = 10  # Initial value of the resource per capita
replenishment_amount = 0.46 #Amount of resources (per capita) that is replenished each time step
coop_consumption_rate = 0.2  # Consumption rates for cooperative individuals (NOTE this includes the proportion of cooperative individuals because it is constant)
freerider_slope = -1
freerider_intercept = 5 #this needs to be set so that the function returns the same cooperative consumption rate when the proportion that know is 0.5)

# Example: Erdos-Renyi graph
graph = nx.erdos_renyi_graph(num_individuals, 0.02)

# Initialize and run the model
model = ABMModel(graph, p_freeriders, beta, mu, T, initial_resource, replenishment_amount, coop_consumption_rate, freerider_slope, freerider_intercept)

# Run the simulation
model.run()

# Get the history of the simulation
history = model.get_history()
resource_history = model.get_resource_history()

# Convert history into time series data for plotting
timesteps = range(T)
DontKnow_proportions = [h['DontKnow'] for h in history]
Know_proportions = [h['Know'] for h in history]

# Create a figure with a grid layout: 2 rows (1 plot on the first row, 2 on the second)
fig = plt.figure(figsize=(20, 15))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # Define grid spec (1st row, centered; 2nd row with two plots)

# First plot: Initial network (spanning the first row)
ax0 = fig.add_subplot(grid[0, :])  # This spans both columns in the first row
ax0.set_title("Initial network config")
pos = nx.spring_layout(graph)  # Layout for the graph
nx.draw(graph, pos, ax=ax0, node_color='gray', with_labels=True, node_size=300, font_size=10)

# Second plot: Proportion of individuals by type
ax1 = fig.add_subplot(grid[1, 0])  # Left plot on second row
ax1.plot(timesteps, DontKnow_proportions, label='Dont Know', marker='o')
ax1.plot(timesteps, Know_proportions, label='Know', marker='o')
if model.trigger_time is not None:
    ax1.axvline(x=model.trigger_time, color='purple', linestyle='--', label=f'Trigger at t={model.trigger_time-1}')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Proportion of Individuals')
ax1.set_title('Proportion of Individuals by Type')
ax1.legend()
ax1.grid(True)

# Third plot: Resource consumption over time
ax2 = fig.add_subplot(grid[1, 1])  # Right plot on second row
ax2.plot(timesteps, resource_history, label='Resource Value', color='blue', marker='o')
if model.trigger_time is not None:
    ax2.axvline(x=model.trigger_time, color='purple', linestyle='--', label=f'Trigger at t={model.trigger_time}')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Resource Value')
ax2.set_title('Resource Value Over Time')
ax2.legend()
ax2.grid(True)

# Adjust layout and show the figure
plt.tight_layout()

#save the figure

# Define the file name structure based on the parameters
file_name = (
    f'sim_ErdosRenyi_p_coop_{p_freeriders}_beta_{beta}_mu_{mu}_T_{T}_'
    f'resource_{initial_resource}_replenishment_{replenishment_amount}.png'
)

# Replace any special characters or spaces with underscores to ensure compatibility
file_name = file_name.replace(".", "_")

# Define the full path where the file will be saved
PATH_TO_SAVE = "../Output/Images/V0/"

# Save the figure using the structured file name
plt.savefig(os.path.join(PATH_TO_SAVE, file_name))

# Show the figure
plt.show()