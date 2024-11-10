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
    def __init__(self, initial_value, replenishment_proportion, consumption_rates, max_capacity):
        """
        Parameters:
        - initial_value: The starting amount of the resource.
        - replenishment_proportion: Proportion of the remaining resource to replenish at each step.
        - consumption_rates: Dictionary defining how much each type of individual consumes.
        """
        self.value = initial_value
        self.replenishment_proportion = replenishment_proportion
        self.consumption_rates = consumption_rates
        self.max_capacity = max_capacity

    def logistic_growth(self, R, r, K):
        return r * R * (1 - R / K)

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


        # Logistic growth of the resource
        
        growth = self.logistic_growth(self.value, self.replenishment_proportion, self.max_capacity)
    
        # Actualizar la cantidad de recurso disponible (después de generación y consumo)
        self.value = self.value + growth - total_consumption


        # Ensure resource value does not go negative
        if self.value < 0:
            self.value = 0


    def get_value(self):
        return self.value


class ABMModel:
    def __init__(self, graph, p_cooperative, beta, mu, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho, max_capacity):
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
        self.max_capacity = max_capacity
        self.triggered = False  # Track if the extra cooperation has been triggered
        self.trigger_time = None  # Time step when the critical threshold is crossed
        self.individuals = {i: Individual(i) for i in self.graph.nodes()}
        self.initialize_individuals()
        self.history = []  # Store the proportions at each time step
        self.resource_history = []  # Store resource values at each step

        # Initialize the Resources class
        self.resources = Resources(initial_resource, replenishment_proportion, consumption_rates, max_capacity)

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
        proportions = {key: state_counts[key] / self.num_individuals for key in state_counts} # 
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

