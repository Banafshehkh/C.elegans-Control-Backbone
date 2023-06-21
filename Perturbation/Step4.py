#Author: Banaafsheh Khazali
#Date: Jun 21, 2023

#import libraries
import time
import os
import random

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy import integrate, signal, sparse, linalg
from threading import Thread
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation, ArtistAnimation
import joblib
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------Simulation Variables--------------------------------------------
# """ Number of Neurons """"
N = 279

# """ Cell membrane conductance (pS) """
Gc = 0.1

# """ Cell Membrane Capacitance """
C = 0.015

# """ Gap Junctions (Electrical, 279*279) """
ggap = 1.0
Gg_Static = np.load('/content/drive/MyDrive/connectome/Gg.npy')

# """ Synaptic connections (Chemical, 279*279) """
gsyn = 1.0
Gs_Static = np.load('/content/drive/MyDrive/connectome/Gs.npy')

# """ Leakage potential (mV) """
Ec = -35.0

# """ Directionality (279*1) """
E = np.load('/content/drive/MyDrive/connectome/emask.npy')
E = -48.0 * E
EMat = np.tile(np.reshape(E, N), (N, 1))

# """ Synaptic Activity Parameters """
ar = 1.0/1.5 # Synaptic activity's rise time
ad = 5.0/1.5 # Synaptic activity's decay time
B = 0.125 # Width of the sigmoid (mv^-1)

# """ Input_Mask/Continuous Transtion """
transit_Mat = np.zeros((2, N))

t_Tracker = 0
Iext = 100000

rate = 0.025
offset = 0.15

t_Switch = 0
transit_End = 10


# """ Connectome Arrays """
Gg_Dynamic = Gg_Static.copy()
Gs_Dynamic = Gs_Static.copy()

# """ Data matrix stack size """
stack_Size = 5000
init_data_Mat = np.zeros((stack_Size + 5000, N))
data_Mat = np.zeros((stack_Size, N))

InMask = np.zeros(N)
oldMask = np.zeros(N)
newMask = np.zeros(N)

# ------------------------------------------------ Connections Initialization -------------------
# """ Determine the length of the binary string based on the total number of connections
total_connections = len(Gg_Static) + len(Gs_Static)
binary_string_length = total_connections

# """ Initialize the population
population_size = 10  
population = []

for _ in range(population_size):
    # Generate a random binary string for an individual in the population
    individual = [random.choice([0, 1]) for _ in range(binary_string_length)]
    population.append(individual)

# --------------------------------------------------- Mask Transition --------------------------
def transit_Mask(input_Array):

    global t_Switch, oldMask, newMask, transit_End, Vth_Static

    transit_Mat[0,:] = transit_Mat[1,:]

    t_Switch = t_Tracker

    transit_Mat[1,:] = input_Array

    oldMask = transit_Mat[0,:]
    newMask = transit_Mat[1,:]

    Vth_Static = EffVth_rhs(Iext, newMask)
    transit_End = t_Switch + 0.3

    print(oldMask, newMask, t_Switch, transit_End)



def update_Mask(old, new, t, tSwitch):

    return np.multiply(old, 0.5-0.5*np.tanh((t-tSwitch)/rate)) + np.multiply(new, 0.5+0.5*np.tanh((t-tSwitch)/rate))

# --------------------------------------------------- Nodal Ablation -----------------------------------------
def modify_Connectome(ablation_Array):

    global Vth_Static, Gg_Dynamic, Gs_Dynamic

    apply_Col = np.tile(ablation_Array, (N, 1))
    apply_Row = np.transpose(apply_Col)

    apply_Mat = np.multiply(apply_Col, apply_Row)

    Gg_Dynamic = np.multiply(Gg_Static, apply_Mat)
    Gs_Dynamic = np.multiply(Gs_Static, apply_Mat)

    try:
        newMask

    except NameError:

        EffVth(Gg_Dynamic, Gs_Dynamic)

        if np.sum(ablation_Array) != N:

            print("Neurons " + str(np.where(ablation_Array == False)[0]) + " are ablated")

        else:

            print("All Neurons healthy")

        print("EffVth Recalculated")

    else:

        EffVth(Gg_Dynamic, Gs_Dynamic)
        Vth_Static = EffVth_rhs(Iext, newMask)

        if np.sum(ablation_Array) != N:

            print("Neurons " + str(np.where(ablation_Array == False)[0]) + " are ablated")

        else:

            print("All Neurons healthy")

        print("EffVth Recalculated")
        print("Vth Recalculated")

# ------------------------------------------------ Efficient V-threshold computation -------------------
def EffVth(Gg, Gs):

    Gcmat = np.multiply(Gc, np.eye(N))
    EcVec = np.multiply(Ec, np.ones((N, 1)))

    M1 = -Gcmat
    b1 = np.multiply(Gc, EcVec)

    Ggap = np.multiply(ggap, Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, N, N).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(gsyn, Gs)
    s_eq = round((ar/(ar + 2 * ad)), 4)
    sjmat = np.multiply(s_eq, np.ones((N, N)))
    S_eq = np.multiply(s_eq, np.ones((N, 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, N, N).toarray()

    b3 = np.dot(Gs_ij, np.multiply(s_eq, E))

    M = M1 + M2 + M3

    global LL, UU, bb

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, N)



def EffVth_rhs(Iext, InMask):

    InputMask = np.multiply(Iext, InMask)
    b = np.subtract(bb, InputMask)

    Vth = linalg.solve_triangular(UU, linalg.solve_triangular(LL, b, lower = True, check_finite=False), check_finite=False)

    return Vth



def voltage_filter(v_vec, vmax, scaler):

    filtered = vmax * np.tanh(scaler * np.divide(v_vec, vmax))

    return filtered

# ---------------------------------------------------------Right hand side -------------------------
def membrane_voltageRHS(t, y):
    global InMask, Vth, t_switch, transit_End

    """ Split the incoming values """
    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(Gc, (Vvec - Ec))

    """ Gg(Vi - Vj) Computation """
    Vrep = np.tile(Vvec, (N, 1))
    GapCon = np.multiply(Gg_Dynamic, np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), EMat)
    SynapCon = np.multiply(np.multiply(Gs_Dynamic, np.tile(SVec, (N, 1))), VsubEj).sum(axis = 1)

    global InMask, Vth

    if t >= t_Switch and t <= transit_End:

        InMask = update_Mask(oldMask, newMask, t, t_Switch + offset)
        Vth = EffVth_rhs(Iext, InMask)

    else:

        InMask = newMask
        Vth = Vth_Static

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(ar, (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, Vth)))))

    SynDrop = np.multiply(ad, SVec)

    """ Input Mask """
    Input = np.multiply(Iext, InMask)

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/C
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))
        


def compute_jacobian(t, y):

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (N, 1))

    J1_M1 = -np.multiply(Gc, np.eye(N))
    Ggap = np.multiply(ggap, Gg_Dynamic)
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag)
    Gsyn = np.multiply(gsyn, Gs_Dynamic)
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / C

    J2_M4_2 = np.subtract(EMat, np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / C

    global InMask, Vth

    if t >= t_Switch and t <= transit_End:

        InMask = update_Mask(oldMask, newMask, t, t_Switch + offset)
        Vth = EffVth_rhs(Iext, InMask)

    else:

        InMask = newMask
        Vth = Vth_Static

    sigmoid_V = np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, Vth))))
    J3_1 = np.multiply(ar, 1 - SVec)
    J3_2 = np.multiply(B, sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-ar, sigmoid_V), ad))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

# ----------------------------------------------------Simulation Initiator ---------------------------------
def run_Network(t_Delta, atol):
    dt = t_Delta
    InitCond = 10**(-4)*np.random.normal(0, 0.94, 2*N)

    """ Configuring the ODE Solver """
    r = integrate.ode(membrane_voltageRHS, compute_jacobian).set_integrator('vode', atol=atol, min_step=dt*1e-6, method='bdf')
    r.set_initial_value(InitCond, 0)

    init_data_Mat[0, :] = InitCond[:N]

    session_Data = []
    oldMask = newMask = np.zeros(N)
    t_Switch = 0
    transit_End = 0.3
    k = 1

    while r.successful() and k < stack_Size + 50:
        r.integrate(r.t + dt)
        data = np.subtract(r.y[:N], Vth)
        init_data_Mat[k, :] = voltage_filter(data, 500, 1)
        t_Tracker = r.t
        k += 1

    # emit('new data', init_data_Mat[50:, :].tolist())
    session_Data.append(np.asarray(init_data_Mat[50:, :].tolist()))
    return session_Data

EffVth(Gg_Static, Gs_Static)

# -------------------------------------------------------- Initialize Population ------------------------
def initialize_population(population_size, chromosome_length):
    population = []

    for _ in range(population_size):
        # Generate a random binary string as an individual in the population
        individual = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(individual)

    return population

# ------------------------------------------------------- Calculate Fitness Score ---------------------
def calculate_fitness_score(y_pred):
    
    fitness_scores = np.where(np.isin(y_pred, 1), 1, 0)
    # print("fitness scores are:", fitness_scores)


    return fitness_scores

# -------------------------------------------------------Evaluate Fitness Function ----------------------------
def evaluate_fitness(population):
    fitness_scores = []
    success_count = 0
    removed_connections = []


    for individual in population:

        t_Delta = 0.001
        atol = 1e-6

        #Neuron Stimulation
        indices = [276, 278]
        value = 9
        newMask[indices] = value


        simulation_data = run_Network(t_Delta, atol)
        simulation_array = np.array(simulation_data[0])
        output_file = "simulation_output.npy"
        np.save(output_file, simulation_array)
        # Flatten the data into a 2D array
        X = simulation_array[:6000,:]
        n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_timesteps * n_features)


        model = joblib.load("/content/drive/MyDrive/connectome/random_forest_model.joblib")
        predictions = model.predict(X_flat.reshape(1, -1))
        print(predictions)
        fitness_score = calculate_fitness_score(predictions)
        fitness_scores.append(fitness_score)

        if np.sum(fitness_score) > 0:
          success_count +=1
          
        success_percentage = (success_count/len(population))*100
        removed_connections.append(np.where(np.array(fitness_score) == 0)[0])
        # print("fitness scores in evaluate fitness", fitness_scores)

    return fitness_scores, success_count, success_percentage

# ------------------------------------------- Parent Selection -------------------
def roulette_wheel_selection(probabilities):
    # Check if all probabilities are zero
    if all(prob == 0 for prob in probabilities):
        return random.randint(0, len(probabilities) - 1)

    # Perform roulette wheel selection
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    random_value = random.uniform(0, sum(probabilities))

    for index, cumulative_prob in enumerate(cumulative_probabilities):
        if random_value <= cumulative_prob:
            return index

    # Fallback to random selection if no valid index is found
    return random.randint(0, len(probabilities) - 1)

def select_parents(population, fitness_scores):
    # Calculate selection probabilities
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]

    # Select two parents using roulette wheel selection
    parent1_index = roulette_wheel_selection(probabilities)
    parent2_index = roulette_wheel_selection(probabilities)

    parent1 = population[parent1_index]
    parent2 = population[parent2_index]

    return parent1, parent2

# ----------------------------------------------------- Crossover -------------------
def crossover(parent1, parent2):
    # Single-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring = parent1[:crossover_point] + parent2[crossover_point:]

    return offspring

# ----------------------------------------------- Mutation ----------------------------
def mutation(individual, mutation_rate):
    # Perform mutation operation on the individual with the given mutation rate
    # Bit flip mutation
    mutated_individual = []
    for gene in individual:
        if random.random() < mutation_rate:
            mutated_gene = 0 if gene == 1 else 1
        else:
            mutated_gene = gene
        mutated_individual.append(mutated_gene)
    print("mutateded individual is:", mutated_individual)

    return mutated_individual

# -------------------------------------------------- Create offspring -----------------
def create_offspring(selected_parents, crossover_rate, mutation_rate):
    offspring = []

    # Perform crossover and mutation for each pair of selected parents
    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]

        # Perform crossover with a certain probability
        if random.random() < crossover_rate:
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
        else:
            child1 = parent1
            child2 = parent2

        # Perform mutation on the offspring with a certain probability
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)

        offspring.append(child1)
        offspring.append(child2)
        print("offspring is:", offspring)

    return offspring

# ------------------------------------------------------ Replace population ------------------
def replace_population(population, offspring):
    num_offspring = len(offspring)
    num_replace = min(num_offspring, len(population))

    # Randomly select individuals from the population to be replaced
    replace_indices = random.sample(range(len(population)), num_replace)

    # Replace selected individuals with the offspring
    for i, replace_index in enumerate(replace_indices):
        population[replace_index] = offspring[i]
    print("population is:", population)

    return population

# -------------------------------------------------------------------------------------------
def desired_fitness_reached(fitness_scores, desired_threshold):
    # Check if any fitness score exceeds the desired threshold
    for score in fitness_scores:
        if score >= desired_threshold:
            return True
    return False
# --------------------------------------------------- Run the Geneic Algorithm ---------------------------------------
def genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate):
    # Step 1: Initialize the population
    population = initialize_population(population_size, chromosome_length)

    for generation in range(num_generations):
        # Step 2: Evaluate the fitness of each individual
        fitness_scores = evaluate_fitness(population)
        # print("fitness score in the loop:", fitness_scores)

        # Step 3: Select parents for reproduction
        selected_parents = select_parents(population, fitness_scores)
        print("selected parents in the loop", selected_parents)

        # Step 4: Create offspring through crossover and mutation
        offspring = create_offspring(selected_parents, crossover_rate, mutation_rate)
        print("offspring in the loop", offspring)

        # Step 5: Replace individuals in the current population with offspring
        population = replace_population(population, offspring)
        print("population", population)

        # Termination condition example: Check if desired fitness threshold is reached
        desired_threshold = 0.7  # Choose your desired threshold
        if desired_fitness_reached(fitness_scores, desired_threshold):
            print("desired fitness reached!")
            break

    return population

# --------------------------------------------------------------------------------
num_generations = 100
crossover_rate = 0.8
mutation_rate = 0.01
chromosome_length = np.count_nonzero(Gg_Dynamic)


population_sizes = [10, 15, 20]

# Create empty lists to store the success percentages for each population size
success_percentages_10 = []
success_percentages_15 = []
success_percentages_20 = []

# Iterate over each population size, calculate the success percentage, and store it in the respective lists
for population_size in population_sizes:
    population = initialize_population(population_size, chromosome_length)
    fitness_scores, success_count, success_percentage = evaluate_fitness(population)

    if population_size == 10:
        success_percentages_10.append(success_percentage)
    elif population_size == 15:
        success_percentages_15.append(success_percentage)
    elif population_size == 20:
        success_percentages_20.append(success_percentage)

    print(f"Population Size: {population_size}, Success Percentage: {success_percentage}")

# Plot the success percentages for each population size on the same figure
plt.plot(population_sizes, success_percentages_10, 'bo-', label='Population Size 10')
plt.plot(population_sizes, success_percentages_15, 'go-', label='Population Size 15')
plt.plot(population_sizes, success_percentages_20, 'ro-', label='Population Size 20')

plt.xlabel('Population Size')
plt.ylabel('Success Percentage')
plt.title('Success Percentage for Different Population Sizes')
plt.legend()
plt.show()

