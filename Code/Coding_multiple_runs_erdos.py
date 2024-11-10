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
        self.graph = graph
        self.num_individuals = graph.number_of_nodes()
        self.p_cooperative = p_cooperative
        self.beta = beta
        self.mu = mu
        self.T = T
        self.rho = rho
        self.critical_value = critical_value
        self.max_capacity = max_capacity
        self.triggered = False
        self.trigger_time = None
        self.individuals = {i: Individual(i) for i in self.graph.nodes()}
        self.initialize_individuals()
        self.history = []
        self.resource_history = []
        self.resource_collapsed = False
        self.collapse_time = None
        self.resources = Resources(initial_resource, replenishment_proportion, consumption_rates, max_capacity)

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
