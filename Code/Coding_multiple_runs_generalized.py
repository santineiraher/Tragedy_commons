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
        self.value = initial_value
        self.replenishment_proportion = replenishment_proportion
        self.consumption_rates = consumption_rates

    def update(self, proportions):
        total_consumption = 0
        for ind_type, proportion in proportions.items():
            total_consumption += self.consumption_rates[ind_type] * proportion
        self.value = self.value - total_consumption + self.replenishment_proportion
        if self.value < 0:
            self.value = 0

    def get_value(self):
        return self.value

class ABMModel:
    def __init__(self, graph, p_cooperative, beta, mu, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho):
        self.graph = graph
        self.num_individuals = graph.number_of_nodes()
        self.p_cooperative = p_cooperative
        self.beta = beta
        self.mu = mu
        self.T = T
        self.rho = rho
        self.critical_value = critical_value
        self.triggered = False
        self.trigger_time = None
        self.individuals = {i: Individual(i) for i in self.graph.nodes()}
        self.initialize_individuals()
        self.history = []
        self.resource_history = []
        self.resource_collapsed = False
        self.collapse_time = None
        self.resources = Resources(initial_resource, replenishment_proportion, consumption_rates)

    def initialize_individuals(self):
        for i in self.individuals:
            self.individuals[i].type = 'Indifferent'
        initial_cooperative = random.sample(list(self.individuals.keys()), int(self.p_cooperative * self.num_individuals))
        for i in initial_cooperative:
            self.individuals[i].type = 'Cooperative'

    def step(self, t):
        if not self.triggered and self.resources.get_value() < self.critical_value:
            self.triggered = True
            self.trigger_time = t

        for i in self.individuals:
            random_value = np.random.random()
            neighbors = list(self.graph.neighbors(i))
            num_cooperative_neighbors = sum([1 for n in neighbors if self.individuals[n].type == 'Cooperative'])
            prob_cooperative = 1 - (1 - self.beta) ** num_cooperative_neighbors
            if self.triggered:
                prob_cooperative += self.rho
            if self.individuals[i].type == 'Indifferent':
                if random_value < prob_cooperative:
                    self.individuals[i].type = 'Cooperative'
            elif self.individuals[i].type == 'Cooperative':
                if random_value < self.mu:
                    self.individuals[i].type = 'Indifferent'

        if self.resources.get_value() <= 0 and self.collapse_time is None:
            self.collapse_time = t

    def run(self):
        for t in range(self.T):
            self.step(t)
            self.record_state()

    def record_state(self):
        state_counts = {'Indifferent': 0, 'Cooperative': 0}
        for individual in self.individuals.values():
            state_counts[individual.type] += 1
        proportions = {key: state_counts[key] / self.num_individuals for key in state_counts}
        self.history.append(proportions)
        self.resources.update(proportions)
        self.resource_history.append(self.resources.get_value())

    def has_collapsed(self):
        return self.resources.get_value() <= 0

    def get_collapse_time(self):
        return self.collapse_time

# Simulation Parameters
num_individuals = 100
p_cooperative = 0.05
beta = 0.8
mu = 0.2
T = 200

# Resource parameters
initial_resource = 10
replenishment_proportion = 0.46
consumption_rates = {'Indifferent': 1, 'Cooperative': 0.1}

# Critical resource parameters
critical_value = 5
rho = 0

def run_single_simulation(graph_type, **kwargs):
    if graph_type == 'erdos_renyi':
        graph = nx.erdos_renyi_graph(kwargs['num_individuals'], kwargs['p'])
    elif graph_type == 'watts_strogatz':
        graph = nx.watts_strogatz_graph(kwargs['num_individuals'], kwargs['k'], kwargs['p'])
    elif graph_type == 'barabasi_albert':
        graph = nx.barabasi_albert_graph(kwargs['num_individuals'], kwargs['m'])

    model = ABMModel(graph, p_cooperative, beta, mu, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho)
    model.run()
    if model.has_collapsed():
        return True, model.get_collapse_time()
    return False, None

def run_simulations(graph_type, num_simulations, **kwargs):
    collapses = 0
    collapse_times = []

    for _ in range(num_simulations):
        collapsed, collapse_time = run_single_simulation(graph_type, **kwargs)
        if collapsed:
            collapses += 1
            collapse_times.append(collapse_time)

    valid_collapse_times = [time for time in collapse_times if time is not None]
    avg_collapse_proportion = collapses / num_simulations
    avg_collapse_time = np.mean(valid_collapse_times) if valid_collapse_times else 200
    return avg_collapse_proportion, avg_collapse_time

# Watts-Strogatz parameters
k = 2  # Each node is connected to k neighbors in the ring topology
p_values = np.linspace(0.01, 0.3, 50)
num_simulations = 100

collapse_proportions_ws = []
avg_collapse_times_ws = []

for p in p_values:
    avg_collapse_proportion, avg_collapse_time = run_simulations('watts_strogatz', num_simulations, num_individuals=num_individuals, k=k, p=p)
    collapse_proportions_ws.append(avg_collapse_proportion)
    avg_collapse_times_ws.append(avg_collapse_time)
    print(f"p = {p:.3f}: Collapse Proportion = {avg_collapse_proportion:.2f}, Avg Collapse Time = {avg_collapse_time}")

# Barabási-Albert parameters
m_values = np.arange(1, 11)
collapse_proportions_ba = []
avg_collapse_times_ba = []

for m in m_values:
    avg_collapse_proportion, avg_collapse_time = run_simulations('barabasi_albert', num_simulations, num_individuals=num_individuals, m=m)
    collapse_proportions_ba.append(avg_collapse_proportion)
    avg_collapse_times_ba.append(avg_collapse_time)
    print(f"m = {m}: Collapse Proportion = {avg_collapse_proportion:.2f}, Avg Collapse Time = {avg_collapse_time}")

# Plotting results for Watts-Strogatz and Barabási-Albert
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Watts-Strogatz plot
ax1.plot(p_values, collapse_proportions_ws, marker='o', label='Watts-Strogatz')
ax1.set_xlabel('p (rewiring probability)')
ax1.set_ylabel('Avg Proportion of Collapsed Runs')
ax1.set_title('Collapse Proportion vs Watts-Strogatz p')
ax1.grid(True)

# Barabási-Albert plot
ax2.plot(m_values, avg_collapse_times_ba, marker='o', label='Barabási-Albert')
ax2.set_xlabel('m (new edges per node)')
ax2.set_ylabel('Avg Time to Collapse')
ax2.set_title('Avg Time to Collapse vs Barabási-Albert m')
ax2.grid(True)

plt.tight_layout()
plt.show()

file_name="collapse_proportion_vs_p_barabasi_watts_strogatz.png"

# Replace any special characters or spaces with underscores to ensure compatibility

# Define the full path where the file will be saved
PATH_TO_SAVE = "../Output/Images/Estructural Parameters/"

# Save the figure using the structured file name
plt.savefig(os.path.join(PATH_TO_SAVE, file_name))