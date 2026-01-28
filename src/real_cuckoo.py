import time
import random
import math
import sys
import os
import datetime
import numpy as np
from tqdm import tqdm

def compute_f(point):
    f = -1 * math.sin(point[0])*math.sqrt(point[0]) * math.sin(point[1])*math.sqrt(point[1]) * \
        math.sin(point[2])*math.sqrt(point[2]) * math.sin(point[3])*math.sqrt(point[3])
    return f

n = 4

min_range = [0, 0, 0, 0]
max_range = [10, 10, 10, 10]

start_time = time.time()

num_cyc = 24000
N = 50 # number of nests
p = 0.6 # fraction of local flights to undertake
q = 0.25 # fraction of nests to abandon
alpha = 1.6 # scaling factor for Levy flights
beta = 1.5

timed = True
max_time = 9.7 # maximum time in seconds

# for Mantegna
sigma = ((math.gamma(1 + beta) * math.sin(math.pi * beta / 2)) / (beta * math.gamma((1 + beta) / 2) * 2 ** ((beta - 1) / 2))) ** (1 / beta)

def fitness(point):
    # minimise f, where f<100
    # fitness must always be > 0
    return 100 + compute_f(point)

def levy_flight(point, alpha):
    # Mantegna's algorithm
    isValid = False
    while not isValid:
        # U sampled from normal dist N(0, sigma)
        U = random.gauss(0, sigma)

        # V sampled from normal dist N(0, 1)
        V = random.gauss(0, 1)

        # Levy flight step
        step = U / abs(V) ** (1 / beta)

        new_point = [point[i] + alpha * step for i in range(n)]
        
        # check if new point is within bounds
        isValid = True
        for i in range(n):
            if new_point[i] < min_range[i] or new_point[i] > max_range[i]:
                isValid = False
                break

    return new_point

def local_flight(point):
    # visit local near neighbours
    isValid = False

    while not isValid:
        new_point = [point[i] + random.uniform(-1, 1) for i in range(n)]

        # check if new point is within bounds
        isValid = True
        for i in range(n):
            if new_point[i] < min_range[i] or new_point[i] > max_range[i]:
                isValid = False
    return new_point

# main function
# cuckoo_search(n, N, num_cyc, p, q, alpha, beta)
def cuckoo_search(N, num_cyc, p, q, alpha):
    start_t = time.time()
    # randomly generate population of nests
    P = []
    for i in range(N):
        nest = [random.uniform(min_range[j], max_range[j]) for j in range(n)]
        P.append([nest, fitness(nest)]) # store nest and its fitness together

    best = min(P, key=lambda x: x[1]) # best nest with lowest fitness
    minima = [best[0]] # store all minima found (NO FITNESS INCLUDED)
    for i in range(1, N):
        if P[i][1] == best[1]:
            minima.append(P[i][0])
        else:
            break

    # main loop
    with tqdm(total=num_cyc, desc="Cuckoo Search", unit="cycle") as pbar:
        for t in range(num_cyc):
            # Update progress bar with current best fitness
            pbar.set_postfix({'best_fitness': f"{best[1]:.6f}", 'time_left': f"{max_time - (time.time() - start_t):.1f}s"})
            pbar.update(1)
            
            # check if any minima can be added
            for i in range(1, N):
                if P[i][1] == best[1]:
                    minima.append(P[i][0])
                else:
                    break

            if timed and time.time() - start_t > max_time:
                pbar.write("Time limit reached")
                return minima

            # levy flights from each nest
            for i in range(N):
                # undertake Levy flight from x_i to y_i
                y = levy_flight(P[i][0], alpha) # y_i
                y_fitness = fitness(y)
                if y_fitness < P[i][1]:
                    # replace x_i with y_i if y_i is better
                    P[i] = [y, y_fitness]

                    # update best
                    if y_fitness < best[1]:
                        best = [y, y_fitness]

            # local search with probability p
            local_flights = random.sample(range(N), int(p * N))
            for j in local_flights:
                # undertake local flight
                y = local_flight(P[j][0])
                y_fitness = fitness(y)
                if y_fitness < P[j][1]:
                    P[j] = [y, y_fitness]

                if timed and time.time() - start_t > max_time:
                    pbar.write("Time limit reached")
                    return minima

            # rank nests by fitness
            P.sort(key=lambda x: x[1])

            # update best_fitness and minima if necessary
            if P[0][1] < best[1]:
                best = P[0]
                
                # restart minima collection
                minima = [best[0]]

            # abandon q fraction of worst nests
            abandoned_nests = P[-int(q * N):]
            for k in abandoned_nests:
                idx = P.index(k)
                # generate new random nest to replace
                y = [random.uniform(min_range[j], max_range[j]) for j in range(n)]
                y = levy_flight(y, alpha) # levy flight from new nest
                P[idx] = [y, fitness(y)]

                # update best_fitness if necessary
                if P[idx][1] < best[1]:
                    best = P[idx]
                    minima = [P[idx][0]] # reset minima
                
                if timed and time.time() - start_t > max_time:
                    pbar.write("Time limit reached")
                    return minima
        
    return minima

minima = cuckoo_search(N, num_cyc, p, q, alpha)
minimum = minima[0]
min_f = compute_f(minimum)

# ---

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)

print("\nYou have found a minimum value of {0} and a minimum point of {1}.".format(min_f, minimum))
print("Your elapsed time was {0} seconds.\n".format(elapsed_time))
    
