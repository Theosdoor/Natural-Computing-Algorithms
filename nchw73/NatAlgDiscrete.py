"""
Natural Algorithm for Discrete Problems: Cuckoo Search for Graph Partitioning
Usage: python NatAlgDiscrete.py <config_file.yaml>
Example: python NatAlgDiscrete.py config_A.yaml
"""

alg_code = "CS"
problem_code = "GP"

import time
import os
import random
import math
import sys
from datetime import datetime
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from utils import location_of_GraphFiles, location_of_witness_set, get_a_timestamp_for_an_output_file

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_the_graph_file(problem_code, graph_digit):
    vertices_tag =          "number of vertices = "
    edges_tag =             "number of edges = "
    colours_to_use_tag =    "number of colours to use = "
    sets_in_partition_tag = "number of partition sets = "
    len_vertices_tag = len(vertices_tag)
    len_edges_tag = len(edges_tag)
    len_colours_to_use_tag = len(colours_to_use_tag)
    len_sets_in_partition_tag = len(sets_in_partition_tag)
    
    input_file = location_of_GraphFiles(problem_code, graph_digit)
    f = open(input_file, 'r')
    whole_line = f.readline()
    vertices = whole_line[len_vertices_tag:len(whole_line) - 1]
    v = int(vertices)
    whole_line = f.readline()
    edges = whole_line[len_edges_tag:len(whole_line) - 1]
    if problem_code == "GC":
        whole_line = f.readline()
        colours_to_use = whole_line[len_colours_to_use_tag:len(whole_line) - 1]
        colours = int(colours_to_use)
    if problem_code == "GP":
        whole_line = f.readline()
        sets_in_partition = whole_line[len_sets_in_partition_tag:len(whole_line) - 1]
        sets_in_partition = int(sets_in_partition)
    matrix = []
    for i in range(0, v - 1):
        whole_line = f.readline()
        if i != v - 2:
            splitline = whole_line.split(',')
            splitline.pop(v - 1 - i)
            for j in range(0, v - 1 - i):
                splitline[j] = int(splitline[j])
        else:
            splitline = whole_line[0:len(whole_line) - 1]
            splitline = [int(splitline)]
        splitline.insert(0, 0)
        matrix.append(splitline[:])            
    matrix.append([0])
    for i in range(0, v):
        for j in range(0, i):
            matrix[i].insert(j, matrix[j][i])
    f.close()

    edges = []
    for i in range(0, v):
        for j in range(i + 1, v):
            if matrix[i][j] == 1:
                edges.append([i, j])

    if problem_code == "GC":
        return v, edges, matrix, colours
    elif problem_code == "GP":
        return v, edges, matrix, sets_in_partition
    else:
        return v, edges, matrix

def main():
    # Load configuration
    if len(sys.argv) < 2:
        print("Usage: python NatAlgDiscrete.py <config_file.yaml>")
        print("Example: python NatAlgDiscrete.py config_A.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    # Extract configuration
    graph_digit = config['graph_digit']
    num_cyc = config['num_cyc']
    N = config['N']
    p = config['p']
    q = config['q']
    alpha = config['alpha']
    beta = config['beta']
    alpha_decay = config['alpha_decay']
    w = config['w']
    p_ranked = config['p_ranked']
    init_greedy = config['init_greedy']
    levy_strat = config['levy_strat']
    neighbour_limit = config['neighbour_limit']
    timed = config['timed']
    max_time = config['max_time']
    
    print(f"Running Cuckoo Search on Graph {graph_digit}")
    print(f"Configuration: {config_path}")
    print(f"Parameters: N={N}, num_cyc={num_cyc}, p={p}, q={q}, alpha={alpha}, beta={beta}")
    
    # Read graph file
    if problem_code == "GC":
        v, edges, matrix, colours = read_the_graph_file(problem_code, graph_digit)
    elif problem_code == "GP":
        v, edges, matrix, sets_in_partition = read_the_graph_file(problem_code, graph_digit)
    else:
        v, edges, matrix = read_the_graph_file(problem_code, graph_digit)

    start_time = time.time()
    
    # Set up derived parameters
    n = v  # if each vertex is a dimension
    alphat = alpha  # Levy scaling that changes with time
    
    # for Mantegna
    sigma_sq = ((math.gamma(1 + beta) * math.sin(math.pi * beta / 2)) / 
                (beta * math.gamma((1 + beta) / 2) * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    
    # for partitioning
    min_size = v // sets_in_partition
    rem = v % sets_in_partition
    max_size = min_size + 1 if rem > 0 else min_size
    
    start_t = time.time()
    
    # Vertex class
    class Vertex:
        '''
        Class to store data for each vertex that won't change during the algorithm.
        Saves computing it again!
        '''
        def __init__(self, id):
            self.id = id
            self.degree = 0
            self.neighbours = []

            # get neighbours
            for i in range(v):
                if matrix[id][i] == 1:
                    self.degree += 1
                    self.neighbours.append(i)
            self.neighbours = tuple(self.neighbours)

        def get_neighbours(self):
            return list(self.neighbours)[:]

    vertices = tuple([Vertex(i) for i in range(v)])

    def get_conflicts(partition):
        '''Get number of conflicts in partition'''
        conflicts = 0
        for x in vertices:
            i = x.id
            for j in x.get_neighbours():
                if partition[i] != partition[j] and j > i:
                    conflicts += 1
        return conflicts

    def get_set_sizes(partition):
        '''Get number of vertices (i.e. size) of each set in the given partition'''
        set_sizes = [0] * sets_in_partition
        for i in range(v):
            set_sizes[partition[i]-1] += 1
        return set_sizes

    def fitness(partition):
        '''Fitness function to minimise number of conflicts. Fitness = 0 is optimal.'''
        return get_conflicts(partition)

    def gen_rand_nest():
        '''Generate a random *balanced* partition'''
        partition = []
        for i in range(1, sets_in_partition+1):
            partition += [i] * min_size

        if rem > 0:
            choices = random.sample(range(1, sets_in_partition+1), rem)
            partition += choices
        
        random.shuffle(partition)
        return partition

    def gen_greedy_nest():
        '''Generate greedy nest (not recommended - typically underperforms random)'''
        partition = [0] * v
        sets_and_sizes = {i: 0 for i in range(1, sets_in_partition+1)}

        vtxs = list(vertices[:])
        random.shuffle(vtxs)

        for vtx in vtxs:
            checklist = list(sets_and_sizes.items())
            checklist.sort(key=lambda x: x[1])

            conflicts_per_set = [float('inf')] * sets_in_partition

            for k in sets_and_sizes.keys():
                conflicts_per_set[k-1] = 0

            while len(checklist) > 0:
                min_set = checklist.pop(0)[0]

                conflicts = 0
                for neighbour in vtx.get_neighbours():
                    if partition[neighbour] == min_set:
                        conflicts += 1
                
                if conflicts == 0:
                    partition[vtx.id] = min_set
                    sets_and_sizes[min_set] += 1

                    if sets_and_sizes[min_set] == max_size:
                        sets_and_sizes.pop(min_set)
                        conflicts_per_set[min_set-1] = float('inf')
                    break
                
                conflicts_per_set[min_set-1] = conflicts

            if partition[vtx.id] == 0:
                min_conflicts = min(conflicts_per_set)
                min_set = conflicts_per_set.index(min_conflicts) + 1
                partition[vtx.id] = min_set
                sets_and_sizes[min_set] += 1

                if sets_and_sizes[min_set] == max_size:
                    sets_and_sizes.pop(min_set)
                    conflicts_per_set[min_set-1] = float('inf')

        if 0 in partition:
            print("Greedy partition failed!")
            sys.exit()

        return partition

    def neighbour_swap(partition, a, b, limit=1):
        '''
        FM-inspired heuristic for Levy flight.
        1. Swap 2 given vertices in partition.
        2. Check neighbours of a, b and swap if beneficial.
        '''
        new = partition[:]
        swaps = 0

        A_part = partition[a]
        B_part = partition[b]

        new[a], new[b] = new[b], new[a]
        swaps += 1

        if limit == 1:
            return new

        a_checklist = []
        b_checklist = []
        for i in vertices[a].get_neighbours():
            if partition[i] == A_part:
                a_checklist.append(i)
        for i in vertices[b].get_neighbours():
            if partition[i] == B_part:
                b_checklist.append(i)

        a_checklist = random.sample(a_checklist, min(neighbour_limit, len(a_checklist)))
        b_checklist = random.sample(b_checklist, min(neighbour_limit, len(b_checklist)))

        A_vertices = []
        B_vertices = []
        A_deg0 = []
        B_deg0 = []

        for i in range(v):
            if partition[i] == A_part:
                if vertices[i].degree == 0:
                    A_deg0.append(i)
                A_vertices.append(i)
            if partition[i] == B_part:
                if vertices[i].degree == 0:
                    B_deg0.append(i)
                B_vertices.append(i)

        while len(a_checklist) != len(b_checklist):
            if len(a_checklist) < len(b_checklist):
                if len(A_deg0) > 0:
                    a_checklist.append(A_deg0.pop())
                else:
                    a_checklist.append(A_vertices.pop())
            else:
                if len(B_deg0) > 0:
                    b_checklist.append(B_deg0.pop())
                else:
                    b_checklist.append(B_vertices.pop())

        initial_conflicts = get_conflicts(new)
      
        while swaps < limit and (len(a_checklist) + len(b_checklist) > 0):
            if len(a_checklist) > 0:
                a = random.choice(a_checklist)
            else:
                a = random.choice
            b = random.choice(b_checklist)

            new[a], new[b] = new[b], new[a]
            new_conflicts = get_conflicts(new)
            gain = initial_conflicts - new_conflicts

            if gain <= 0:
                new[a], new[b] = new[b], new[a]
                a_checklist.remove(a)
                b_checklist.remove(b)
                continue

            swaps += 1
            initial_conflicts = new_conflicts

            a_checklist.remove(a)
            b_checklist.remove(b)

        return new

    def levy_flight(partition, alpha):
        '''Levy Flight using Mantegna's algorithm'''
        new = partition[:]

        U = random.gauss(0, sigma_sq)
        V = random.gauss(0, 1)

        step = abs(U / abs(V) ** (1 / beta))
        M = int(alpha * step)
        
        if levy_strat == 'R':
            for _ in range(M):
                u = random.randint(0, n-1)
                v_idx = random.randint(0, n-1)
                while new[u] == new[v_idx]:
                    v_idx = random.randint(0, n-1)
                new[u], new[v_idx] = new[v_idx], new[u]
        elif levy_strat == 'FM':
            u = random.randint(0, n-1)
            v_idx = random.randint(0, n-1)
            while new[u] == new[v_idx]:
                v_idx = random.randint(0, n-1)
            new = neighbour_swap(new, u, v_idx, M)
        else:
            print("Invalid Levy strategy! Try 'FM' or 'R'")
            sys.exit()

        return new

    def local_flight(partition):
        '''Local search - visit local near neighbours'''
        new = partition[:]

        u = random.randint(0, n-1)
        v_idx = random.randint(0, n-1)
        while new[u] == new[v_idx]:
            v_idx = random.randint(0, n-1)
        new[u], new[v_idx] = new[v_idx], new[u]
        return new

    def cuckoo_search(N, num_cyc, p, q):
        '''Cuckoo Search algorithm'''
        nonlocal alphat, alpha

        P = []
        for _ in range(N):
            if init_greedy:
                nest = gen_greedy_nest()
            else:
                nest = gen_rand_nest()
            P.append([nest, fitness(nest)])

        best = min(P, key=lambda x: x[1])

        # Progress bar
        with tqdm(total=num_cyc, desc="Cuckoo Search", unit="cycle") as pbar:
            for t in range(num_cyc):
                if timed and time.time() - start_t > max_time:
                    pbar.write(f"Time limit reached at cycle {t}")
                    return best[0]
                
                # Update progress bar with current best fitness
                pbar.set_postfix({'best_conflicts': best[1], 'time_left': f"{int(max_time - (time.time() - start_t))}s"})
                pbar.update(1)
               
                # Levy flights from each nest
                for i in range(N):
                    y = levy_flight(P[i][0], alpha)
                    y_fitness = fitness(y)
                    if y_fitness < P[i][1]:
                        P[i] = [y, y_fitness]

                        if y_fitness < best[1]:
                            best = [y, y_fitness]

                # Local search with probability p
                local_flights = random.sample(range(N), int(p * N))
                for j in local_flights:
                    y = local_flight(P[j][0])
                    y_fitness = fitness(y)
                    if y_fitness < P[j][1]:
                        P[j] = [y, y_fitness]

                    if timed and time.time() - start_t > max_time:
                        pbar.write(f"Time limit reached at cycle {t}")
                        return best[0]

                # Rank nests by fitness
                P.sort(key=lambda x: x[1])
                ranked_nests = P[:w]

                if P[0][1] < best[1]:
                    best = P[0]

                    if best[1] == 0:
                        pbar.write(f"Optimal solution found at cycle {t}!")
                        return best[0]
                
                if alpha_decay:
                    alphat = alphat / math.sqrt(t+1)

                # Abandon q fraction of worst nests & replace
                abandoned_nests = P[-int(q * N):]
                for k in abandoned_nests:
                    idx = P.index(k)
                    if w>0 and random.random() < p_ranked:
                        y = random.choices(ranked_nests, weights=[1/nest[1] for nest in ranked_nests])[0][0]
                        y = levy_flight(y, alphat)
                    else:
                        y = gen_rand_nest()

                    P[idx] = [y, fitness(y)]

                    if P[idx][1] < best[1]:
                        best = P[idx]

                        if best[1] == 0:
                            pbar.write(f"Optimal solution found at cycle {t}!")
                            return best[0]
                    
                    if timed and time.time() - start_t > max_time:
                        pbar.write(f"Time limit reached at cycle {t}")
                        return best[0]
        
        return best[0]

    partition = cuckoo_search(N, num_cyc, p, q)
    conflicts = get_conflicts(partition)

    print(f'\nFinal conflicts: {conflicts}')

    now_time = time.time()
    elapsed_time = round(now_time - start_time, 1)

    error_flag = False

    if not problem_code in ["GC", "GP", "CL"]:
        print("*** error: 'problem_code' = {0} is illegal".format(problem_code))
        error_flag = True

    if problem_code == "GP":
        if type(conflicts) != int:
            print("*** error: 'conflicts' is not an integer: it is {0} and it has type {1})".format(conflicts, type(conflicts)))
            error_flag = True
        elif conflicts < 0:
            print("*** error: 'conflicts' should be non-negative where it is {0}".format(conflicts))
            error_flag = True
        elif type(partition) != list:
            print("*** error: 'partition' is not a list (it has type {0})".format(type(partition)))
            error_flag = True
        elif len(partition) != v:
            print("*** error: 'partition' is a list of length {0} whereas it should have length {1}".format(len(partition), v))
            error_flag = True
        else:
            for i in range(0, v):
                if type(partition[i]) != int:
                    print("*** error: 'partition[{0}]' = {1} is not an integer (it has type {2})".format(i, partition[i], type(partition[i])))
                    error_flag = True
                    break
                elif partition[i] < 1 or partition[i] > sets_in_partition:
                    print("*** error: 'partition[{0}]' = {1} which is a bad partite set (partite set names must range from 1 up to 'sets_in_partition' = {2})".format(i, partition[i], sets_in_partition))
                    error_flag = True
                    break
        if error_flag == False:
            true_conflicts = 0
            for i in range(0, v):
                for j in range(i + 1, v):
                    if matrix[i][j] == 1 and partition[i] != partition[j]:
                        true_conflicts = true_conflicts + 1
            if conflicts != true_conflicts:
                print("*** error: you claim {0} but there are actually {1} conflicts\n".format(conflicts, true_conflicts))
                error_flag = True

    if not alg_code in ["AB", "FF", "CS", "WO", "BA"]:
        print("*** error: 'alg_code' = {0} is invalid".format(alg_code))
        error_flag = True

    if type(n) != int:
        print("*** error: 'n' is not an integer: it is {0} and it has type {1})".format(n, type(n)))
    if type(num_cyc) != int:
        print("*** error: 'num_cyc' is not an integer: it is {0} and it has type {1})".format(num_cyc, type(num_cyc)))

    if alg_code == "CS":
        if type(N) != int:
            print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
            error_flag = True
        if type(p) != int and type(p) != float:
            print("*** error: 'p' is not an integer or a float: it is {0} and it has type {1})".format(p, type(p)))
            error_flag = True
        if type(q) != int and type(q) != float:
            print("*** error: 'q' is not an integer or a float: it is {0} and it has type {1})".format(q, type(q)))
            error_flag = True
        if type(alpha) != int and type(alpha) != float:
            print("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1})".format(alpha, type(alpha)))
            error_flag = True
        if type(beta) != int and type(beta) != float:
            print("*** error: 'beta' is not an integer or a float: it is {0} and it has type {1})".format(beta, type(beta)))
            error_flag = True

    if error_flag == False:
        timestamp = get_a_timestamp_for_an_output_file()
        witness_set = location_of_witness_set(graph_digit, timestamp)

        f = open(witness_set, "w")

        f.write("problem code = {0}\n".format(problem_code))
        f.write("graph = {0}Graph{1}.txt with (|V|,|E|) = ({2},{3})\n".format(problem_code, graph_digit, v, len(edges)))
        f.write("number of partition sets = {0}\n".format(sets_in_partition))
        f.write("algorithm code = {0}\n".format(alg_code))
        f.write("associated parameters [n, num_cyc, N, p, q, alpha, beta] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}, {5}, {6}]\n".format(n, num_cyc, N, p, q, alpha, beta))
        f.write("conflicts = {0}\n".format(conflicts))
        f.write("elapsed time = {0}\n".format(elapsed_time))
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y-%H:%M:%S")
        f.write("date-time = {0}\n".format(date_time))
        
        alg_number = ord(alg_code[0]) + ord(alg_code[1])
        len_date_time = len(date_time)
        date_time_number = 0
        for i in range(0, len_date_time):
            date_time_number = date_time_number + ord(date_time[i])
        
        diff = abs(partition[0] - partition[v - 1])
        for i in range(0, v - 1):
            diff = diff + abs(partition[i + 1] - partition[i])
        
        certificate = user_number + alg_number + date_time_number + diff
        f.write("certificate = {0}\n".format(certificate))

        for i in range(0, v):
            f.write("{0},".format(partition[i]))
            if (i + 1) % 40 == 0:
                f.write("\n")
        if v % 40 != 0:
            f.write("\n")

        f.close()
            
        print("witness file 'Witness{0}_{1}.txt' saved".format(graph_digit, timestamp))

    else:
        print("\n*** ERRORS: the witness file has not been saved - fix your errors first!")

if __name__ == "__main__":
    main()
