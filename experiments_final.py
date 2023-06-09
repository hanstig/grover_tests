import numpy as np
import time
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
from qiskit.tools.jupyter import *
from qiskit.providers.fake_provider import FakeOslo
from qiskit.tools.parallel import parallel_map
from qiskit.providers.aer.noise import NoiseModel




#
# Helper functions
#

# r is the number of grover iterations, m is the number of marked items, n is the size of the database
def grover_success_prob(r, m, n):
    return np.sin((2*r + 1) * np.arcsin(np.sqrt(m/n)))**2 * 100

# n is the size of the database, m is the number of marked items
def grover_iterations(m, n):
    return int(np.round(np.pi/(4 * np.arcsin(np.sqrt(m/n)))-1/2))

#selects n unique random numbers from the in the range [0 - upper_bound)
def select_random_elements(upper_bound, n):
    elements = list(range(upper_bound))
    selected = []
    for i in range(n):
        index = np.random.randint(len(elements))
        selected.append(elements[index])
        del elements[index]
    return selected

# Checks if the integer representetion of binary_string exists in number_list
def binary_string_in_list(number_list, binary_string):

    num = int(binary_string, 2)

    for number in number_list:
        if num == number:
            return True

    return False    

    # for number in number_list:
    #     number_string = "{0:b}".format(number).zfill(len(binary_string))
    #     if (len(binary_string) != len(number_string)):
    #         raise ValueError(
    #             'Something went wrong when converting a number to binary string')
    #     if binary_string == number_string:
    #         return True
    # return False

#
# Grover implementation
#

def phase_oracle(n, indicies_to_mark, name='Oracle'):
    qc = QuantumCircuit(n, name=name)
    oracle_matrix = np.identity(2**n)
    for index_to_mark in indicies_to_mark:
        oracle_matrix[index_to_mark, index_to_mark] = -1
    qc.unitary(Operator(oracle_matrix), range(n))
    return qc


# Dynamic programming optimization of diffuser function
# dict to save dicts that have been computed 
diffuser_dict = {}

def diffuser(n):
    if n not in diffuser_dict:
        qc = QuantumCircuit(n, name='Diffuser')
        qc.h(range(n))
        qc.append(phase_oracle(n, [0]), range(n))
        qc.h(range(n))
        diffuser_dict.update({n: qc})
    return diffuser_dict.get(n)


def Grover(n, marked, r):
    qc = QuantumCircuit(n, n)

    qc.h(range(n))
    qc_phase = phase_oracle(n, marked)
    for _ in range(r):
        qc.append(qc_phase, range(n))
        qc.append(diffuser(n), range(n))
    qc.measure(range(n), range(n))
    return qc

#
# Functions to run experiments
#

# Runs noiseless experiment with n qubits
def simulator_experiment(n):
    start = time.time()
    backend = Aer.get_backend('aer_simulator')
    
    nums = range(1, 2**n + 1)

    # runs experiment in parallel
    results = parallel_map(experiment_parallel, nums, task_args=(
        n, backend), task_kwargs={}, num_processes=4)

    end = time.time()
    
    with open(f"./results/noiseless_experiment_{n}qubits.txt", "a") as f:
        print("*---------------", file = f)
        print(f"Results for {n} qubits. time = {end - start} seconds\n", file = f)
        
        print("actual probabilities:\n[", end = "", file = f)
        for result in results:
            print(result[0], end=", ", file = f)
        print("]\n", file = f)

        print("expected probabilities:\n[", end = "", file = f)
        for result in results:
            print(result[1], end=", ", file = f)
        print("]", file = f)
        print("*---------------\n", file = f)
        

# helper function to parallelize simulator_experiment
# nums - how many marked items should be selected
# n - number of qubits
# backend - backend for the simulator
def experiment_parallel(nums, n, backend):
    print(f"started job {nums}")

    iterations = grover_iterations(nums, 2**n)
    expected_success_probability = grover_success_prob(iterations, nums, 2**n)
    iterations_per_marked_item_set = 10
    shots = 10000 

    hits = 0
    for j in range(iterations_per_marked_item_set):
        marked = select_random_elements(2**n, nums)
        qc = Grover(n, marked, iterations)
        result = execute(qc, backend, shots=shots, memory=True).result()
        counts = result.get_memory(qc)
        for count in counts:
            found_correct_element = binary_string_in_list(marked, count)
            if found_correct_element:
                hits += 1
        print(f">>> Job {nums} finished iteration {j + 1} / {iterations_per_marked_item_set}")
        
    return (((hits/(iterations_per_marked_item_set * shots)) * 100), expected_success_probability)

def simulator_experiment_noise(n):
    start = time.time()

    nums = range(1, 2**n + 1)

    results = parallel_map(experiment_parallel_noise, nums, task_args=([n]), task_kwargs={}, num_processes=8)

    end = time.time()

    with open(f"./results/noise_experiment_{n}qubits.txt", "a") as f:
        print("*---------------", file = f)
        print(f"Results for {n} qubits. time = {end - start} seconds\n", file = f)
        
        print("actual probabilities:\n[", end = "", file = f)
        for result in results:
            print(result[0], end=", ", file = f)
        print("]\n", file = f)

        print("expected probabilities:[", end = "", file = f)
        for result in results:
            print(result[1], end = ", ", file  = f)
        print("]", file = f)
        print("*---------------\n", file = f)

def experiment_parallel_noise(nums, n):
    print(f"started job {nums}")

    # how many grover iterations should be performed
    iterations = grover_iterations(nums, 2**n)
    expected_success_probability = grover_success_prob(iterations, nums, 2**n)

    noisemodel = NoiseModel.from_backend(FakeOslo())
    sim = AerSimulator(noise_model = noisemodel)
    iterations_per_marked_item_set = 5
    shots = 2000

    hits = 0
    for j in range(iterations_per_marked_item_set):
        marked = select_random_elements(2**n, nums)
        qc = Grover(n, marked, iterations)
        
        t_qc = transpile(qc, sim, optimization_level=0)

        result = sim.run(t_qc, shots=shots, memory=True).result()

        counts = result.get_memory()

        for count in counts:
            found_correct_element = binary_string_in_list(marked, count)
            if found_correct_element:
                hits += 1
        print(f">>> Job {nums} finished iteration {j + 1} / {iterations_per_marked_item_set}")

    return (((hits/(iterations_per_marked_item_set * shots)) * 100) , expected_success_probability)

simulator_experiment(5)
simulator_experiment(7)

simulator_experiment_noise(5)
simulator_experiment_noise(7)

