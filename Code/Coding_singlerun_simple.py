import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os

class Individual:
    def __init__(self, node_id, individual_type='Indifferent'):
        self.node_id = node_id
        self.type = individual_type  # Indifferent or Cooperative


class Resources:
    def __init__(self, initial_value, replenishment_proportion, consumption_rates):
        """
        Parameters:
        - initial_value: The starting amount of the resource.
        - replenishment_proportion: Proportion of the remaining resource to replenish at each step.
        - consumption_rates: Dictionary defining how much each type of individual consumes.
        """
        self.value = initial_value
        self.replenishment_proportion = replenishment_proportion
        self.consumption_rates = consumption_rates

    def update(self, proportions):
        """
        Update the resource value based on consumption and replenishment.

        Parameters:
        - proportions: A dictionary with the proportion of individuals in each type (Cooperative, Indifferent).
        """
        total_consumption = 0

        # Calculate total consumption based on the proportion of population and individual consumption rates
        for ind_type, proportion in proportions.items():
            total_consumption += self.consumption_rates[ind_type] * proportion

        # Replenishment is a proportion of the current resource level
        self.value = self.value - total_consumption + self.replenishment_proportion

        # Ensure resource value does not go negative
        if self.value < 0:
            self.value = 0


    def get_value(self):
        return self.value


class ABMModel:
    def __init__(self, graph, p_cooperative, beta, mu, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho):
        """
        Initializes the model with a given graph and other parameters.

        Parameters:
        - graph: The networkx graph to be used in the simulation.
        - p_cooperative: Initial fraction of cooperative individuals.
        - beta: Probability parameter for cooperation.
        - mu: Constant probability of becoming indifferent.
        - T: Number of time steps to simulate.
        - initial_resource: Initial value of the resource pool.
        - replenishment_proportion: Proportion of resources replenished at each step.
        - consumption_rates: Dictionary of consumption rates for each individual type.
        - critical_value: Critical resource value where cooperation probability is boosted.
        - rho: Extra cooperation probability if resources drop below the critical value.
        """
        self.graph = graph
        self.num_individuals = graph.number_of_nodes()
        self.p_cooperative = p_cooperative
        self.beta = beta
        self.mu = mu  # Replaces gamma with mu
        self.T = T
        self.rho = rho  # Extra cooperation probability when resources drop below critical value
        self.critical_value = critical_value
        self.triggered = False  # Track if the extra cooperation has been triggered
        self.trigger_time = None  # Time step when the critical threshold is crossed
        self.individuals = {i: Individual(i) for i in self.graph.nodes()}
        self.initialize_individuals()
        self.history = []  # Store the proportions at each time step
        self.resource_history = []  # Store resource values at each step

        # Initialize the Resources class
        self.resources = Resources(initial_resource, replenishment_proportion, consumption_rates)

    def initialize_individuals(self):
        """Initialize individuals, all are Indifferent and a fraction p are Cooperative"""
        for i in self.individuals:
            self.individuals[i].type = 'Indifferent'

        # Set initial cooperative individuals based on fraction p_cooperative
        initial_cooperative = random.sample(list(self.individuals.keys()), int(self.p_cooperative * self.num_individuals))
        for i in initial_cooperative:
            self.individuals[i].type = 'Cooperative'

    def step(self, t):
        """Perform a single step of the simulation."""
        # Check if the resource has fallen below the critical value and trigger the extra cooperation probability
        if not self.triggered and self.resources.get_value() < self.critical_value:
            self.triggered = True
            self.trigger_time = t  # Record the time when the event happens

        for i in self.individuals:
            random_value = np.random.random()
            neighbors = list(self.graph.neighbors(i))
            num_cooperative_neighbors = sum([1 for n in neighbors if self.individuals[n].type == 'Cooperative'])

            # Probability to become cooperative
            prob_cooperative = 1 - (1 - self.beta) ** num_cooperative_neighbors

            # If the critical event has been triggered, add the extra cooperation probability rho
            if self.triggered:
                prob_cooperative += self.rho

            # Apply state changes based on probabilities
            if self.individuals[i].type == 'Indifferent':
                if random_value < prob_cooperative:
                    self.individuals[i].type = 'Cooperative'
            elif self.individuals[i].type == 'Cooperative':
                if random_value < self.mu:  # Use mu instead of gamma
                    self.individuals[i].type = 'Indifferent'

    def run(self):
        """Run the simulation for T steps."""
        for t in range(self.T):
            self.step(t)
            self.record_state()

    def record_state(self):
        """Record the proportion of each type of individual and the resource value."""
        state_counts = {'Indifferent': 0, 'Cooperative': 0}
        for individual in self.individuals.values():
            state_counts[individual.type] += 1

        # Normalize to get proportions
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
p_cooperative = 0.05  # Initial fraction of cooperative individuals
beta = 0.8  # Probability parameter for cooperation
mu = 0.2  # Constant probability of becoming indifferent
T = 200  # Number of time steps

# Resource parameters
initial_resource = 10  # Initial value of the resource
replenishment_proportion = 0.46  # 1% of the remaining resource is replenished each step
consumption_rates = {'Indifferent': 1, 'Cooperative': 0.1}  # Consumption rates for each type



# Critical resource parameters
critical_value = 5 # If the resource drops below this value, extra cooperation probability is triggered
rho = 0.1  # Extra cooperation probability when the critical resource threshold is crossed

# Example: Erdos-Renyi graph
graph = nx.erdos_renyi_graph(num_individuals, 0.02)

# Initialize and run the model
model = ABMModel(graph, p_cooperative, beta, mu, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho)

# Run the simulation
model.run()

# Get the history of the simulation
history = model.get_history()
resource_history = model.get_resource_history()

# Convert history into time series data for plotting
timesteps = range(T)
indifferent_proportions = [h['Indifferent'] for h in history]
cooperative_proportions = [h['Cooperative'] for h in history]

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
ax1.plot(timesteps, indifferent_proportions, label='Indifferent', marker='o')
ax1.plot(timesteps, cooperative_proportions, label='Cooperative', marker='o')
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
    f'sim_ErdosRenyi_p_coop_{p_cooperative}_beta_{beta}_mu_{mu}_T_{T}_'
    f'resource_{initial_resource}_replenishment_{replenishment_proportion}_'
    f'critical_{critical_value}_rho_{rho}.png'
)

# Replace any special characters or spaces with underscores to ensure compatibility
file_name = file_name.replace(".", "_")

# Define the full path where the file will be saved
PATH_TO_SAVE = "../Output/Images/V0/"

# Save the figure using the structured file name
plt.savefig(os.path.join(PATH_TO_SAVE, file_name))

# Show the figure
plt.show()
