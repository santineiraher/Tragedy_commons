using LightGraphs
using Random
using PyPlot
using StatsBase
using Parameters

function abpath()
    replace(@__DIR__, "Code" => "")
end

function control_indifferentcooperators(numindv, transitionprob, initialprop, time)
    data=zeros(time)
    data2=zeros(time)
    data[1]=numindv*(1-initialprop)
    data2[1]=numindv*initialprop
    for t in 1:time-1
        data[t+1]=data[t]*(1-(transitionprob/numindv))
        data2[t+1]=numindv-data[t+1]
    end
    return hcat(data, data2)
end

let 
    data = control_indifferentcooperators(100, 0.5, 0.05,200)
    test = figure()
    plot(1.0:1.0:200.0,data[:,1], color="blue", label="Indifferent")
    plot(1.0:1.0:200.0,data[:,2], color="orange", label="Cooperative")
    xlabel("Time", fontsize=15)
    ylabel("# of Individuals", fontsize=15)
    legend(fontsize=12)
    # return test
    savefig(joinpath(abpath(), "Output/Images/deterministic_individuals_figure.png"))
end

function resource_regen(δi, δc, Ni, Nc)
    return δi*Ni + δc*Nc
end

function find_time(ind_coop_data, indifferentequalprop)
    numindv = ind_coop_data[1,1]+ind_coop_data[1,2]
    for i in 1:length(ind_coop_data[:,1])
        if isapprox(ind_coop_data[i,1], numindv*indifferentequalprop, atol=0.05)
            return i
        else
        end
    end
end

@with_kw mutable struct Res_Use_Par
    δi = 0.6
    δc = 0.2
end

function control_resources(ind_coop_data, res_regen, indifferentequalprop, res_use_par) 
    @unpack δi, δc = res_use_par
    endtime = find_time(ind_coop_data, indifferentequalprop)
    resourceuse_data = zeros(endtime)
    for i in 1:endtime
        resourceuse_data[i] = res_regen-(δi*ind_coop_data[i,1])-(δc*ind_coop_data[i,2])
    end
    return abs(sum(resourceuse_data))
end

function check_control(numindv, transitionprob, initialprop, time, indifferentequalprop, res_use_par)
    @unpack δi, δc = res_use_par
    res_regen = resource_regen(δi, δc, numindv*indifferentequalprop, numindv*(1-indifferentequalprop))
    ind_coop_data=control_indifferentcooperators(numindv, transitionprob, initialprop, time)
    init_res = control_resources(ind_coop_data, res_regen, indifferentequalprop, res_use_par) 
    resources=zeros(time)
    resources[1]=init_res
    for t in 1:time-1
        resources[t+1]=resources[t]+res_regen-(δi*ind_coop_data[t,1])-(δc*ind_coop_data[t,2])
    end
    return resources
end    

let
    data = check_control(100, 0.5, 0.05, 200, 0.5, Res_Use_Par())
    test = figure()
    plot(1.0:1.0:200.0,data)
    xlabel("Time", fontsize=15)
    ylabel("Resources (units)", fontsize=15)
    ylim(0.0,1040)
    # return test
    savefig(joinpath(abpath(), "Output/Images/deterministic_resources_figure.png"))
end


struct Individual
    node_id::Int
    individual_type::String
end

struct Resources
    value::Float64
    replenishment_proportion::Float64
    consumption_rates::Dict{String, Float64}
end

struct ABMModel
    graph::SimpleGraph
    individuals::Dict{Int, Individual}
    resources::Resources
    p_cooperative::Float64
    beta::Float64
    mu::Float64
    T::Int
    rho::Float64
    critical_value::Float64
    triggered::Bool
    trigger_time::Union{Nothing, Int}
    history::Vector{Dict{String, Float64}}
    resource_history::Vector{Float64}
end

function ABMModel(graph::SimpleGraph, p_cooperative::Float64, beta::Float64, mu::Float64, T::Int,
    initial_resource::Float64, replenishment_proportion::Float64, consumption_rates::Dict{String, Float64},
    critical_value::Float64, rho::Float64)
individuals = Dict(i => Individual(i, "Indifferent") for i in vertices(graph))
# Initialize the population
for i in sample(collect(keys(individuals)), Int(p_cooperative * nv(graph)), replace=false)
individuals[i].individual_type = "Cooperative"
end
resources = Resources(initial_resource, replenishment_proportion, consumption_rates)
ABMModel(graph, individuals, resources, p_cooperative, beta, mu, T, rho, critical_value, false, nothing, [], [])
end

function update!(self::Resources, proportions::Dict{String, Float64})
    total_consumption = 0.0
    # Calculate total consumption based on proportions and consumption rates
    for (ind_type, proportion) in proportions
        total_consumption += self.consumption_rates[ind_type] * proportion
    end

    # Replenish and update resource value
    self.value = self.value - total_consumption + self.replenishment_proportion

    # Ensure resource value is not negative
    if self.value < 0
        self.value = 0
    end
end

function step!(model::ABMModel, t::Int)
    # Trigger extra cooperation if resources fall below the critical value
    if !model.triggered && model.resources.value < model.critical_value
        model.triggered = true
        model.trigger_time = t
    end

    for (i, individual) in model.individuals
        random_value = rand()
        neighbors = neighbors(model.graph, i)
        num_cooperative_neighbors = count(n -> model.individuals[n].individual_type == "Cooperative", neighbors)

        # Probability to become cooperative
        prob_cooperative = 1 - (1 - model.beta) ^ num_cooperative_neighbors

        # Add extra cooperation probability if triggered
        if model.triggered
            prob_cooperative += model.rho
        end

        if individual.individual_type == "Indifferent" && random_value < prob_cooperative
            individual.individual_type = "Cooperative"
        elseif individual.individual_type == "Cooperative" && random_value < model.mu
            individual.individual_type = "Indifferent"
        end
    end
end

function record_state!(model::ABMModel)
    state_counts = Dict("Indifferent" => 0, "Cooperative" => 0)
    for individual in values(model.individuals)
        state_counts[individual.individual_type] += 1
    end

    proportions = Dict(key => count / nv(model.graph) for (key, count) in state_counts)
    push!(model.history, proportions)

    # Update resource based on the proportion of individuals
    update!(model.resources, proportions)
    push!(model.resource_history, model.resources.value)
end

function run!(model::ABMModel)
    for t in 1:model.T
        step!(model, t)
        record_state!(model)
    end
end

# Simulation Parameters
num_individuals = 100
p_cooperative = 0.05
beta = 0.5
mu = 0.2
T = 200

# Resource Parameters
initial_resource = 10.0
replenishment_proportion = 0.46
consumption_rates = Dict("Indifferent" => 1.0, "Cooperative" => 0.1)

# Critical resource parameters
critical_value = 5.0
rho = 0.1

# Create a random graph
graph = SimpleGraph(num_individuals)
for _ in 1:(num_individuals * 0.02 * num_individuals)
    u, v = rand(1:num_individuals), rand(1:num_individuals)
    if u != v
        add_edge!(graph, u, v)
    end
end

# Initialize the model
model = ABMModel(graph, p_cooperative, beta, mu, T, initial_resource, replenishment_proportion, consumption_rates, critical_value, rho)

# Run the simulation
run!(model)

# Extract history
history = model.history
resource_history = model.resource_history

timesteps = 1:T
indifferent_proportions = [h["Indifferent"] for h in history]
cooperative_proportions = [h["Cooperative"] for h in history]

# Plotting
plot(timesteps, indifferent_proportions, label="Indifferent", lw=2, marker=:o)
plot!(timesteps, cooperative_proportions, label="Cooperative", lw=2, marker=:o)
plot!(timesteps, resource_history, label="Resource Value", lw=2, marker=:o, yaxis=:right)
