import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os

class Individual:
    def __init__(self, node_id, individual_type='Indifferent'):
        self.node_id = node_id
        self.type = individual_type  # Indifferent, Cooperative, or Freerider


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

    def update(self, population_counts):
        """
        Update the resource value based on consumption and replenishment.

        Parameters:
        - population_counts: A dictionary with the number of individuals in each type (Cooperative, Indifferent, Freerider).
        """
        total_consumption = 0

        # Calculate total consumption based on population and individual consumption rates
        for ind_type, count in population_counts.items():
            total_consumption += self.consumption_rates[ind_type] * count

        # Ensure resource value does not go negative
        if self.value < 0:
            self.value = 0


        #TO-DO: Make it constant

        # Replenish a proportion of the remaining resource
        replenishment_amount = self.replenishment_proportion * self.value

        # Update resource value by subtracting total consumption
        self.value = self.value - total_consumption
        self.value += replenishment_amount

    def get_value(self):
        return self.value


class ABMModel:
    def __init__(self, graph, p_cooperative, beta, gamma, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho):
        """
        Initializes the model with a given graph and other parameters.

        Parameters:
        - graph: The networkx graph to be used in the simulation.
        - p_cooperative: Initial fraction of cooperative individuals.
        - beta: Probability parameter for cooperation.
        - gamma: Probability of becoming a Freerider.
        - T: Number of time steps to simulate.
        - initial_resource: Initial value of the resource pool.
        - replenishment_proportion: Proportion of resources replenished at each step.
        - consumption_rates: Dictionary of consumption rates for each individual type.
        - critical_value: Critical resource value where cooperation probability is boosted.
        - rho: Extra cooperation probability if resources exceed the critical value.
        """
        self.graph = graph
        self.num_individuals = graph.number_of_nodes()  # Automatically use the number of nodes in the graph
        self.p_cooperative = p_cooperative
        self.beta = beta
        self.gamma = gamma
        self.T = T
        self.rho = rho  # Extra cooperation probability
        self.critical_value = critical_value  # Critical resource threshold
        self.triggered = False  # To check if extra cooperation has been triggered
        self.trigger_time = None  # Time step when the critical threshold is crossed
        self.individuals = {i: Individual(i) for i in self.graph.nodes()}
        self.initialize_individuals()
        self.history = []  # List to store the proportions at each time step
        self.resource_history = []  # List to store resource values at each step

        # Initialize the Resources class
        self.resources = Resources(initial_resource, replenishment_proportion, consumption_rates)

    def initialize_individuals(self):
        """Initialize individuals, all are Indifferent and a fraction p are Cooperative"""
        for i in self.individuals:
            self.individuals[i].type = 'Indifferent'

        # Set initial cooperative individuals based on fraction p_cooperative
        initial_cooperative = random.sample(list(self.individuals.keys()),
                                            int(self.p_cooperative * self.num_individuals))
        for i in initial_cooperative:
            self.individuals[i].type = 'Cooperative'

    def step(self, t):
        """Perform a single step of the simulation."""
        # Check if the resource has exceeded the critical value and trigger the extra cooperation probability
        if not self.triggered and self.resources.get_value() < self.critical_value:
            self.triggered = True
            self.trigger_time = t  # Record the time when the event happens

        # For this step, we iterate over indifferent individuals and apply the rules
        for i in self.individuals:
            random_value = np.random.random()
            neighbors = list(self.graph.neighbors(i))
            num_cooperative_neighbors = sum([1 for n in neighbors if self.individuals[n].type == 'Cooperative'])
            # Probability to become cooperative
            prob_cooperative = 1 - (1 - self.beta) ** num_cooperative_neighbors

            # If the critical event has been triggered, add the extra cooperation probability rho
            if self.triggered:
                prob_cooperative += self.rho

            prob_freerider = self.gamma

            if self.individuals[i].type == 'Indifferent':
                if random_value < prob_cooperative:
                    self.individuals[i].type = 'Cooperative'
                elif random_value < prob_cooperative + prob_freerider:
                    self.individuals[i].type = 'Freerider'
                else:
                    self.individuals[i].type = 'Indifferent'
            elif self.individuals[i].type == 'Cooperative':
                if random_value < self.gamma:
                    self.individuals[i].type = 'Freerider'
                else:
                    self.individuals[i].type = 'Cooperative'
            else:
                if random_value < prob_cooperative:
                    self.individuals[i].type = 'Cooperative'
                else:
                    self.individuals[i].type = 'Freerider'

    def run(self):
        """Run the simulation for T steps."""
        for t in range(self.T):
            self.step(t)
            self.record_state()

    def record_state(self):
        """Record the proportion of each type of individual and the resource value."""
        state_counts = {'Indifferent': 0, 'Cooperative': 0, 'Freerider': 0}
        for individual in self.individuals.values():
            state_counts[individual.type] += 1

        # Normalize to get proportions
        proportions = {key: state_counts[key] / self.num_individuals for key in state_counts}
        self.history.append(proportions)

        # Update the resource pool
        self.resources.update(state_counts)
        self.resource_history.append(self.resources.get_value())

    def get_state(self):
        """Return the current state of the individuals."""
        state = {i: self.individuals[i].type for i in self.individuals}
        return state

    def get_history(self):
        """Return the history of proportions at each time step."""
        return self.history

    def get_resource_history(self):
        """Return the history of resource values."""
        return self.resource_history

    def draw_graph(self, title):
        """Draw the graph with node colors representing their type."""
        # Get node colors based on individual type
        color_map = {
            'Indifferent': 'gray',
            'Cooperative': 'green',
            'Freerider': 'red'
        }

        node_colors = [color_map[self.individuals[node].type] for node in self.graph.nodes()]

        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(self.graph)  # Layout for the graph
        nx.draw(self.graph, pos, node_color=node_colors, with_labels=True, node_size=300, font_size=10)
        plt.title(title)
        plt.show()


# Simulation Parameters
num_individuals = 100  # Number of individuals in the graph
p_cooperative = 0.05  # Initial fraction of cooperative individuals
beta = 0.05  # Probability parameter for cooperation
gamma = 0.02  # Probability of becoming a Freerider
T = 100  # Number of time steps

# Resource parameters
initial_resource = 1000  # Initial value of the resource
replenishment_proportion = 0.01  # 1% of the remaining resource is replenished each step
consumption_rates = {'Indifferent': 0.2, 'Cooperative': 0.15, 'Freerider': 0.25}  # Consumption rates for each type

# Critical resource parameters
critical_value = 600  # If the resource exceeds this value, extra cooperation probability is triggered
rho = 0.1  # Extra cooperation probability when the critical resource threshold is crossed

# Example: Erdos-Renyi graph
graph = nx.erdos_renyi_graph(num_individuals, 0.01)

# Initialize and run the model
model = ABMModel(graph, p_cooperative, beta, gamma, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho)

# Visualize the graph before the simulation (initial state)
model.draw_graph('Initial State of the Graph')

# Run the simulation
model.run()

# Get the history of the simulation
history = model.get_history()
resource_history = model.get_resource_history()

# Convert history into time series data for plotting
timesteps = range(T)
indifferent_proportions = [h['Indifferent'] for h in history]
cooperative_proportions = [h['Cooperative'] for h in history]
freerider_proportions = [h['Freerider'] for h in history]


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
ax1.plot(timesteps, freerider_proportions, label='Freerider', marker='o')
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
print(os.getcwd())
PATH_TO_SAVE= "../Output/Images/"
plt.savefig(PATH_TO_SAVE+f'sim_ErdosRenyi3'+'.png')

plt.show()