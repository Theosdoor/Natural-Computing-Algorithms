#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the skeleton program 'NatAlgDiscrete.py' around which you should build your implementation.

# On input 'GCGraphA.txt', say, the output is a witness set that is in the file 'WitnessA_<username>_<timestamp>.txt' where
# '<timestamp>' is a timestamp so that you do not overwrite previously produced witnesses. You can always
# rename these files. However, apart from renaming them you should not tamper with them in any other way.

# It is assumed that all graph files are in a folder called 'GraphFiles' that lies in the same folder as
# this program.

# As regards the four values to be entered below
# - make sure that the all four values appear within double quotes
# - make sure that 'username' is lower-case
# - make sure that no comments are inserted after you have entered the values.

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "nchw73"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "CS"

#################################################################
#### ENTER THE CODE FOR THE GRAPH PROBLEM YOU ARE OPTIMIZING ####
#################################################################

problem_code = "GP"

#############################################################
#### ENTER THE DIGIT OF THE INPUT GRAPH FILE (A, B OR C) ####
#############################################################

graph_digit = "A"

################################################################
#### DO NOT TOUCH ANYTHING BELOW UNTIL I TELL YOU TO DO SO! ####
####      THIS INCLUDES IMPORTING ADDITIONAL MODULES!       ####
################################################################

import time
import os
import random
import math
import sys
from datetime import datetime

def location_of_GraphFiles(problem_code, graph_digit):
    input_file = os.path.join("GraphFiles", problem_code + "Graph" + graph_digit + ".txt")
    return input_file

def location_of_witness_set(username, graph_digit, timestamp):
    witness_set = "Witness" + graph_digit + "_" + username + "_" + timestamp + ".txt"
    return witness_set

def get_a_timestamp_for_an_output_file():
    local_time = time.asctime(time.localtime(time.time()))
    timestamp = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
    timestamp = timestamp.replace(" ", "0") 
    return timestamp

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
 
if problem_code == "GC":
    v, edges, matrix, colours = read_the_graph_file(problem_code, graph_digit)
elif problem_code == "GP":
    v, edges, matrix, sets_in_partition = read_the_graph_file(problem_code, graph_digit)
else:
    v, edges, matrix = read_the_graph_file(problem_code, graph_digit)

start_time = time.time()

#########################################################################################
#### YOU SHOULDN'T HAVE TOUCHED *ANYTHING* UP UNTIL NOW APART FROM SUPPLYING VALUES  ####
#### FOR 'username', 'alg_code', 'problem_code' and 'graph_digit' AS REQUESTED ABOVE ####
####                        NOW READ THE FOLLOWING CAREFULLY!                        ####
#########################################################################################

# FOR ALL OF THE RESERVED VARIABLES BELOW, YOU MUST ENSURE THAT ON TERMINATION THE TYPE
# OF THE RESPECTIVE VARIABLE IS AS SHOWN.

# For the problem GC, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph                                  int
#   - 'edges' = a list of the edges of the graph (just in case you need them)    list
#   - 'matrix' = the full adjacency matrix of the graph                          list
#   - 'colours' = the maximum number of colours to be used when colouring        int

# For the problem CL, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph                                  int
#   - 'edges' = a list of the edges of the graph (just in case you need them)    list
#   - 'matrix' = the full adjacency matrix of the graph                          list

# For the problem GP, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph                                  int
#   - 'edges' = a list of the edges of the graph (just in case you need them)    list
#   - 'matrix' = the full adjacency matrix of the graph                          list
#   - 'sets_in_partition' = the number of sets in any partition                  int

# These are reserved variables and need to be treated as such, i.e., use these names for these
# concepts and don't re-use the names.

# For the problem GC, you will produce a colouring in the form of a list of v integers called
# 'colouring' where the entries range from 1 to 'colours'. Note! 0 is disallowed as a colour!
# You will also produce an integer in the variable 'conflicts' which denotes how many edges
# are such that the two incident vertices are identically coloured (of course, your aim is to
# MINIMIZE the value of 'conflicts').

# For the problem CL, you will produce a clique in the form of a list of v integers called
# 'clique' where the entries are either 0 or 1. If 'clique[i]' = 1 then this denotes that the
# vertex i is in the clique with 'clique[i]' = 0 denoting otherwise.
# You will also produce an integer in the variable 'clique_size' which denotes how many vertices
# are in your clique (of course, your aim is to MAXIMIZE the value of 'clique_size').

# For the problem GP, you will produce a partition in the form of a list of v integers called
# 'partition' where the entries are in {1, 2, ..., 'sets_in_partition'}. Note! 0 is not the
# name of a partition set! If 'partition[i]' = j then this denotes that the vertex i is in the
# partition set j.
# You will also produce an integer in the variable 'conflicts' which denotes how many edges are
# incident with vertices in different partition sets (of course, your aim is to MINIMIZE the
# value of 'conflicts').

# In consequence, the following additional variables are reserved:
#   - 'colouring'                       list of int
#   - 'conflicts'                       int
#   - 'clique'                          list of int
#   - 'clique_size'                     int
#   - 'partition'                       list of int

# The various algorithms all have additional parameters (see the lectures). These parameters
# are detailed below and are referred to using the following reserved variables.
#
# AB (Artificial Bee Colony)
#   - 'n' = dimension of the optimization problem       int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of employed bees / food sources      int
#   - 'M' = number of onlooker bees                     int
#   - 'lambbda' = limit threshold                       float or int
#
# FF (Firefly)
#   - 'n' = dimension of the optimization problem       int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of fireflies                         int
#   - 'lambbda' = light absorption coefficient          float or int
#   - 'alpha' = scaling parameter                       float or int
#
# CS (Cuckoo Search)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of nests                             int
#   - 'p' = fraction of local flights to undertake      float or int
#   - 'q' = fraction of nests to abandon                float or int
#   - 'alpha' = scaling factor for Levy flights         float or int
#   - 'beta' = parameter for Mantegna's algorithm       float or int
#
# WO (Whale Optimization)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of whales                            int
#   - 'b' = spiral constant                             float or int
#
# BA (Bat)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of bats                              int
#   - 'sigma' = scaling factor                          float or int
#   - 'f_min' = best_colouring frequency                       float or int
#   - 'f_max' = maximum frequency                       float or int

# These are reserved variables and need to be treated as such, i.e., use these names for these
# parameters and don't re-use the names. Don't forget to ensure that on termination all the above
# variables have the stated type. In particular, if you use specific numpy types then you'll need
# to ensure that they are changed prior to termination.

# INITIALIZE THE ACTUAL PARAMETERS YOU USE FOR YOUR ALGORITHM BELOW. ENSURE THAT YOU INITIALIZE
# *ALL* OF THE PARAMETERS REQUIRED APPROPRIATELY (SEE ABOVE) FOR YOUR CHOSEN ALGORITHM.

# In particular, if you are implementing, say, Artificial Bee Colony and you choose to encode
# your bees as tuples of length v (where v is the number of vertices in the input graph) then
# you must initialize n as v (of course, you might have some other encoding where n is different
# to v and if so then you would initialize n accordingly.)

# Also, you may introduce additional parameters if you wish (below) but they won't get written
# to the output file.

# In summary, before you input the bulk of your code, ensure that you:
# - import any (legal) modules you wish to use in the space provided below 
# - initialize your parameters in the space provided below
# - ensure that reserved variables have the correct type on termination.

###########################################
#### NOW YOU CAN ENTER YOUR CODE BELOW ####
###########################################
####################################################
#### FIRST IMPORT ANY MODULES IMMEDIATELY BELOW ####
####################################################

import numpy as np

##########################################################
#### NOW INITIALIZE YOUR PARAMETERS IMMEDIATELY BELOW ####
##########################################################
# BASIC parameters
n = v # if each vertex is a dimension
num_cyc = 100000
N = 50 # number of nests
p = 0.5 # fraction of local flights to undertake
q = 0.25 # fraction of nests to abandon
alpha = 2 # scaling factor for Levy flights
beta = 1.5 # parameter for Mantegna's algorithm

# ENHANCED parameters
alpha_decay = True # if True, then alphat decays over time
alphat = alpha # Levy scaling that changes with time (for Levy flights from new nest)

w = 6 # number of ranked nests (these influence newly generated nests)
p_ranked = 0.8 # probability of picking a ranked nest to generate a new nest via levy flight (alternative is random)

init_greedy = False # if True, then the initial nests are generated using my 'greedy' algorithm. Be warned - they are crap! if False, init nests randomly.

levy_strat = 'R' # 'R' for randomly swapping 2 vertices (basic), 'FM' for (my adaptation of) the Fiduccia-Mattheyses heuristic (enhanced)
neighbour_limit = 4 # maximum number of neighbours to check in flight

# for timing
timed = True
max_time = 58 # maximum time in seconds (60s)

###########################################
#### NOW INCLUDE THE REST OF YOUR CODE ####
###########################################
# NON-EDITABLE parameters
# for Mantegna
sigma_sq = ((math.gamma(1 + beta) * math.sin(math.pi * beta / 2)) / (beta * math.gamma((1 + beta) / 2) * 2 ** ((beta - 1) / 2))) ** (1 / beta)

# for partitioning
min_size = v // sets_in_partition # minimum size for a partition set
rem = v % sets_in_partition # remainder vertices (to add to random partitions)
if rem == 0:
    max_size = min_size
else:
    max_size = min_size + 1

start_t = time.time()

# the code!
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
        self.neighbours = tuple(self.neighbours) # make immutable

    def get_neighbours(self):
        return list(self.neighbours)[:] # return copy so we can't edit it

vertices = tuple([Vertex(i) for i in range(v)]) # store vertex data to refer to throughout (tuple so it cannot be edited)

def get_conflicts(partition):
    '''
    Get number of conflicts in partition
    '''
    conflicts = 0
    for x in vertices:
        i = x.id
        for j in x.get_neighbours():
            if partition[i] != partition[j] and j > i: # check j > i to avoid double counting!
                conflicts += 1
    return conflicts

def get_set_sizes(partition):
    '''
    Get number of vertices (i.e. size) of each set in the given partition
    '''
    set_sizes = [0] * sets_in_partition
    for i in range(v):
        set_sizes[partition[i]-1] += 1 # NOTE: partition sets are 1-indexed, hence -1 in index
    return set_sizes

def fitness(partition):
    '''
    Fitness function to minimise number of conflcits

    Fitness = 0 is optimal.
    '''
    fitness = get_conflicts(partition)
    return fitness

def gen_rand_nest():
    '''
    Generate a random *balanced* partition
    '''
    partition = []
    # ensure all partitions are at least min_size
    for i in range(1, sets_in_partition+1):
        partition += [i] * min_size

    if rem > 0:
        # place remaining vertices in random partitions (max 1 per partition!)
        choices = random.sample(range(1, sets_in_partition+1), rem)
        partition += choices
    
    random.shuffle(partition) # shuffle to randomise which vertices are in which partition
    return partition

def gen_greedy_nest():
    '''
    generate greedy nest.

    1. shuffle vertices
    2. for each vertex, add to smallest partition that doesn't conflict
    3. if all partitions conflict, add to partition with least conflicts (that isn't full!)
    '''
    partition = [0] * v # init partition (invalid since = 0!)
    sets_and_sizes = {i: 0 for i in range(1, sets_in_partition+1)} # init with all partitions and 0 size

    # shuffle vertices
    vtxs = list(vertices[:])
    random.shuffle(vtxs)

    # add to partitions
    for vtx in vtxs:
        # check if can add to partition, starting with smallest
        
        # sort sets by size
        checklist = list(sets_and_sizes.items())
        checklist.sort(key=lambda x: x[1]) # sort by size (ascending)

        conflicts_per_set = [float('inf')] * sets_in_partition

        for k in sets_and_sizes.keys():
            conflicts_per_set[k-1] = 0 # init to 0

        while len(checklist) > 0:
            min_set = checklist.pop(0)[0]

            # check if can add to partition
            conflicts = 0
            for neighbour in vtx.get_neighbours():
                if partition[neighbour] == min_set:
                    conflicts += 1
            # if no conflicts - add it and move to next vertex!
            if conflicts == 0:
                # add to partition
                partition[vtx.id] = min_set
                sets_and_sizes[min_set] += 1

                # remove maxxed out sets
                if sets_and_sizes[min_set] == max_size:
                    sets_and_sizes.pop(min_set)
                    conflicts_per_set[min_set-1] = float('inf') # remove from conflicts list
                break
            
            # otherwise, store conflicts for each set
            conflicts_per_set[min_set-1] = conflicts

        # if still not added, add to partition with least conflicts
        if partition[vtx.id] == 0:
            min_conflicts = min(conflicts_per_set)
            min_set = conflicts_per_set.index(min_conflicts) + 1
            partition[vtx.id] = min_set
            sets_and_sizes[min_set] += 1

            # remove maxxed out sets
            if sets_and_sizes[min_set] == max_size:
                sets_and_sizes.pop(min_set)
                conflicts_per_set[min_set-1] = float('inf') # remove from conflicts list

    if 0 in partition:
        print("Greedy partition failed!")
        sys.exit()

    return partition

def neighbour_swap(partition, a, b, limit=1):
    '''
    My heuristic for Levy flight.
    Inspired by Fiduccia-Mattheyses heuristic as described here: https://web.eecs.umich.edu/~imarkov/pubs/book/b001.pdf.

    1. Swap 2 given vertices in partition.
    2. Check neighbours of a, b and swap if beneficial.

    limit is the maximum number of swaps allowed.
    '''
    new = partition[:]
    swaps = 0

    # define the 2 partitions we're working with
    A_part = partition[a]
    B_part = partition[b]

    # swap a, b
    new[a], new[b] = new[b], new[a]
    swaps += 1

    if limit == 1:
        return new # only 1 swap needed

    # check neighbours in same partition
    a_checklist = []
    b_checklist = []
    for i in vertices[a].get_neighbours():
        if partition[i] == A_part:
            a_checklist.append(i)
    for i in vertices[b].get_neighbours():
        if partition[i] == B_part:
            b_checklist.append(i)

    # take limited number of neighbours to consider as specified
    a_checklist = random.sample(a_checklist, min(neighbour_limit, len(a_checklist)))
    b_checklist = random.sample(b_checklist, min(neighbour_limit, len(b_checklist)))

    A_vertices = [] # get list of all vertices in each partition
    B_vertices = []

    # get v with degree 0 in A and B (swaps are free! - no conflicts)
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


    # make lists the same length by padding with other v in partition
    while len(a_checklist) != len(b_checklist):
        if len(a_checklist) < len(b_checklist):
            if len(A_deg0) > 0:
                # add v with deg 0 if possible
                a_checklist.append(A_deg0.pop())
            else:
                # otherwise add random v from A
                a_checklist.append(A_vertices.pop())
        else:
            if len(B_deg0) > 0:
                b_checklist.append(B_deg0.pop())
            else:
                b_checklist.append(B_vertices.pop())

    initial_conflicts = get_conflicts(new)
  
    # while swaps availble, check neighbours and swap if beneficial
    while swaps < limit and (len(a_checklist) + len(b_checklist) > 0):
        # pick two more vertices to swap
        if len(a_checklist) > 0:
            a = random.choice(a_checklist)
        else:
            a = random.choice
        b = random.choice(b_checklist)

        # does it imporve the partition?
        new[a], new[b] = new[b], new[a]
        new_conflicts = get_conflicts(new)
        gain = initial_conflicts - new_conflicts

        if gain <= 0: # i.e. no improvement
            # Revert move and pick new a,b
            new[a], new[b] = new[b], new[a]
            a_checklist.remove(a)
            b_checklist.remove(b)
            continue

        # otherwise, keep the swap
        swaps += 1
        initial_conflicts = new_conflicts

        # update checklists
        a_checklist.remove(a)
        b_checklist.remove(b)

    return new


def levy_flight(partition, alpha):
    '''
    Levy Flight using Mantegna's algorithm.

    2 strategies:
    1. Randomly swap 2 vertices M times (where M is from Levy dist).
    2. Allow up to M heuristic neighbour swaps
    '''
    new = partition[:]

    # U sampled from normal dist N(0, sigma_sq)
    U = random.gauss(0, sigma_sq)

    # V sampled from normal dist N(0, 1)
    V = random.gauss(0, 1)

    # Levy flight - pick M vertices using levy dist
    step = abs(U / abs(V) ** (1 / beta))
    M = int(alpha * step)
    
    # do levy strategy
    if levy_strat == 'R':
        # swap M random pairs of vertices
        for _ in range(M):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            while new[u] == new[v]: # make sure we're swapping vertices between different partitions
                v = random.randint(0, n-1)
            new[u], new[v] = new[v], new[u]
    elif levy_strat == 'FM':
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        while new[u] == new[v]: # make sure we're swapping vertices between different partitions
            v = random.randint(0, n-1)
        new = neighbour_swap(new, u, v, M)
    else:
        print("Invalid Levy strategy! Try 'FM' or 'R'")
        sys.exit()

    return new

def local_flight(partition):
    '''
    Local search - visit local near neighbours.

    1 Strategy:
    1. Swap random pair of vertices. (Maintains set sizes)
    '''
    new = partition[:]

    # swap random pair of vertices
    u = random.randint(0, n-1)
    v = random.randint(0, n-1)
    while new[u] == new[v]: # make sure we're swapping vertices between different partitions
        v = random.randint(0, n-1)
    new[u], new[v] = new[v], new[u]
    return new
                    

# main function
def cuckoo_search(N, num_cyc, p, q):
    '''
    Cuckoo Search algorithm

    cuckoo_search(n, N, num_cyc, p, q, alpha, beta)

    Note that each population member is stored as [nest, fitness].

    1. Generate random initial population of N nests
    2. Enter main loop:
        a. Generate new nests using Levy flights
        b. Perform local search on a fraction of nests
        c. Replace worst nests with new random nests
    '''
    global beta
    global alphat
    global alpha

    # generate population of random nests
    P = []
    for _ in range(N): # for each nest
        if init_greedy:
            nest = gen_greedy_nest()
        else:
            nest = gen_rand_nest()
        P.append([nest, fitness(nest)]) # store nest and its fitness together

    # track best nest with minimum fitness
    best = min(P, key=lambda x: x[1])

    # main loop
    for t in range(num_cyc):
        if timed and time.time() - start_t > max_time:
            # print("Time limit reached")
            return best[0]
        
        # print updates
        if t%200 == 0:
            print("\nCycle {0}, best_fitness: {1}".format(t, best[1]))
            if timed:
                print("Time remaining: {0}s".format(int(max_time - (time.time() - start_t))))
           
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
        local_flights = random.sample(range(N), int(p * N)) # select p fraction of nests
        for j in local_flights:
            # undertake local flight
            y = local_flight(P[j][0])
            y_fitness = fitness(y)
            if y_fitness < P[j][1]:
                P[j] = [y, y_fitness]

            if timed and time.time() - start_t > max_time:
                # print("Time limit reached")
                return best[0]

        # rank nests by fitness and select w best
        P.sort(key=lambda x: x[1])
        ranked_nests = P[:w]

        # update best if necessary
        # (P[0] is the best partition currently in P)
        if P[0][1] < best[1]:
            best = P[0]

            if best[1] == 0: # optimal solution found
                return best[0]
        
        if alpha_decay:
            alphat = alphat / math.sqrt(t+1) # decrease alpha over time

        # abandon q fraction of worst nests & replace
        abandoned_nests = P[-int(q * N):]
        for k in abandoned_nests:
            idx = P.index(k)
            if w>0 and random.random() < p_ranked:
                # pick one of ranked nests with weight according to fitness
                y = random.choices(ranked_nests, weights=[1/nest[1] for nest in ranked_nests])[0][0]
                y = levy_flight(y, alphat) # levy flight from best nest
            else:
                # generate new random nest to replace abandoned one
                y = gen_rand_nest()

            P[idx] = [y, fitness(y)]

            # update best if necessary
            if P[idx][1] < best[1]:
                best = P[idx]

                if best[1] == 0:
                    return best[0]
            
            if timed and time.time() - start_t > max_time:
                # print("Time limit reached")
                return best[0]
        
    return best[0]

partition = cuckoo_search(N, num_cyc, p, q)
conflicts = get_conflicts(partition) # I get conflicts here, just in case fitness != number of conflicts!

print('Final conflicts:', conflicts)

#########################################################
#### YOU SHOULD HAVE NOW FINISHED ENTERING YOUR CODE ####
####     DO NOT TOUCH ANYTHING BELOW THIS COMMENT    ####
#########################################################

# At this point in the execution, you should have computed
# - the list 'colouring' and integer 'conflicts', if you are solving GC;
# - the list 'clique' and the integer 'clique_size', if you are solving CL; or
# - the list 'partition' and the integer 'conflicts', if you are solving GP.

# What follows is error-checking code. Your output file will be saved only if
# no errors are thrown. If there are errors, you'll be told what they are.

# ANY OUTPUT WILL NOT BE SAVED TO FILE UNTIL NO ERRORS ARE THROWN!

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)

error_flag = False

if not problem_code in ["GC", "GP", "CL"]:
    print("*** error: 'problem_code' = {0} is illegal".format(problem_code))
    error_flag = True

if problem_code == "GC":
    if type(conflicts) != int:
        print("*** error: 'conflicts' is not an integer: it is {0} and it has type {1})".format(conflicts, type(conflicts)))
        error_flag = True
    elif conflicts < 0:
        print("*** error: 'conflicts' should be non-negative whereas it is {0}".format(conflicts))
        error_flag = True
    elif type(colouring) != list:
        print("*** error: 'colouring' is not a list (it has type {0})".format(type(colouring)))
        error_flag = True
    elif len(colouring) != v:
        print("*** error: 'colouring' is a list of length {0} whereas it should have length {1}".format(len(colouring), v))
        error_flag = True
    else:
        for i in range(0, v):
            if type(colouring[i]) != int:
                print("*** error: 'colouring[{0}]' = {1} is not an integer (it has type {2})".format(i, colouring[i], type(colouring[i])))
                error_flag = True
                break
            elif colouring[i] < 1 or colouring[i] > colours:
                print("*** error: 'colouring[{0}]' = {1} which is a bad colour (colours must range from 1 up to 'colours' = {2})".format(i, colouring[i], colours))
                error_flag = True
                break
    if error_flag == False:
        true_conflicts = 0
        for i in range(0, v):
            for j in range(i + 1, v):
                if matrix[i][j] == 1 and colouring[i] == colouring[j]:
                    true_conflicts = true_conflicts + 1
        if conflicts != true_conflicts:
            print("*** error: you claim {0} but there are actually {1} conflicts\n".format(conflicts, true_conflicts))
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

if problem_code == "CL":
    if type(clique_size) != int:
        print("*** error: 'clique_size' is not an integer: it is {0} and it has type {1})".format(clique_size, type(clique_size)))
        error_flag = True
    elif clique_size < 0:
        print("*** error: 'clique_size' should be non-negative where it is {0}".format(clique_size))
        error_flag = True
    elif type(clique) != list:
        print("*** error: 'clique' is not a list (it has type {0})".format(type(clique)))
        error_flag = True
    elif len(clique) != v:
        print("*** error: 'clique' is a list of length {0} whereas it should have length {1}".format(len(clique), v))
        error_flag = True
    else:
        for i in range(0, v):
            if type(clique[i]) != int:
                print("*** error: 'clique[{0}]' = {1} is not an integer (it has type {2})".format(i, clique[i], type(clique[i])))
                error_flag = True
                break
            elif clique[i] < 0 or clique[i] > 1:
                print("*** error: 'clique[{0}]' = {1} where as all entries should be 0 or 1".format(i, colouring[i]))
                error_flag = True
                break
    if error_flag == False:
        true_size = 0
        for i in range(0, v):
            if clique[i] == 1:
                true_size = true_size + 1
        if clique_size != true_size:
            print("*** error: you claim a clique of size {0} but the list contains {1} ones\n".format(clique_size, true_size))
            error_flag = True
        else:
            bad_edges = 0
            bad_clique = False
            for i in range(0, v):
                for j in range(i + 1, v):
                    if clique[i] == 1 and clique[j] == 1 and matrix[i][j] != 1:
                        bad_edges = bad_edges + 1
                        bad_clique = True
            if bad_clique == True:
                print("*** error: the clique of claimed size {0} is not a clique as there are {1} missing edges\n".format(clique_size, bad_edges))
                error_flag = True

if not alg_code in ["AB", "FF", "CS", "WO", "BA"]:
    print("*** error: 'alg_code' = {0} is invalid".format(alg_code))
    error_flag = True

if type(n) != int:
    print("*** error: 'n' is not an integer: it is {0} and it has type {1})".format(n, type(n)))
if type(num_cyc) != int:
    print("*** error: 'num_cyc' is not an integer: it is {0} and it has type {1})".format(num_cyc, type(num_cyc)))

if alg_code == "AB":
    if type(N) != int:
        print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
        error_flag = True
    if type(M) != int:
        print("*** error: 'M' is not an integer: it is {0} and it has type {1})".format(M, type(M)))
        error_flag = True
    if type(lambbda) != int and type(lambbda) != float:
        print("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1})".format(lambbda, type(lambbda)))
        error_flag = True

if alg_code == "FF":
    if type(N) != int:
        print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
        error_flag = True
    if type(lambbda) != int and type(lambbda) != float:
        print("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1})".format(lambbda, type(lambbda)))
        error_flag = True
    if type(alpha) != int and type(alpha) != float:
        print("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1})".format(alpha, type(alpha)))
        error_flag = True

if alg_code == "CS":
    if  type(N) != int:
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

if alg_code == "WO":
    if type(N) != int:
        print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
        error_flag = True
    if type(b) != int and type(b) != float:
        print("*** error: 'b' is not an integer or a float: it is {0} and it has type {1})".format(b, type(b)))
        error_flag = True

if alg_code == "BA":
    if type(sigma) != int and type(sigma) != float:
        print("*** error: 'sigma' is not an integer or a float: it is {0} and it has type {1})".format(sigma, type(sigma)))
        error_flag = True
    if type(f_min) != int and type(f_min) != float:
        print("*** error: 'f_min' is not an integer or a float: it is {0} and it has type {1})".format(f_min, type(f_min)))
        error_flag = True
    if type(f_max) != int and type(f_max) != float:
        print("*** error: 'f_max' is not an integer or a float: it is {0} and it has type {1})".format(f_max, type(f_max)))
        error_flag = True

if error_flag == False:
    
    timestamp = get_a_timestamp_for_an_output_file()
    witness_set = location_of_witness_set(username, graph_digit, timestamp)

    f = open(witness_set, "w")

    f.write("username = {0}\n".format(username))
    f.write("problem code = {0}\n".format(problem_code))
    f.write("graph = {0}Graph{1}.txt with (|V|,|E|) = ({2},{3})\n".format(problem_code, graph_digit, v, len(edges)))
    if problem_code == "GC":
        f.write("colours to use = {0}\n".format(colours))
    elif problem_code == "GP":
        f.write("number of partition sets = {0}\n".format(sets_in_partition))
    f.write("algorithm code = {0}\n".format(alg_code))
    if alg_code == "AB":
        f.write("associated parameters [n, num_cyc, N, M, lambbda] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n,num_cyc, N, M, lambbda))
    elif alg_code == "FF":
        f.write("associated parameters [n, num_cyc, N, lambbda, alpha] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n, num_cyc, N, lambbda, alpha))
    elif alg_code == "CS":
        f.write("associated parameters [n, num_cyc, N, p, q, alpha, beta] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}, {5}, {6}]\n".format(n, num_cyc, N, p, q, alpha, beta))
    elif alg_code == "WO":
        f.write("associated parameters [n, num_cyc, N, b] = ")
        f.write("[{0}, {1}, {2}, {3}]\n".format(n, num_cyc, N, b))
    elif alg_code == "BA":
        f.write("associated parameters [n, num_cyc, sigma, f_max, f_min] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n, num_cyc, sigma, f_max, f_min))
    if problem_code == "GC" or problem_code == "GP":
        f.write("conflicts = {0}\n".format(conflicts))
    elif problem_code == "CL":
        f.write("clique size = {0}\n".format(clique_size))
    f.write("elapsed time = {0}\n".format(elapsed_time))
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y-%H:%M:%S")
    f.write("date-time = {0}\n".format(date_time))
    
    len_username = len(username)
    user_number = 0
    for i in range(0, len_username):
        user_number = user_number + ord(username[i])
    alg_number = ord(alg_code[0]) + ord(alg_code[1])
    len_date_time = len(date_time)
    date_time_number = 0
    for i in range(0, len_date_time):
        date_time_number = date_time_number + ord(date_time[i])
    if problem_code == "GC":
        diff = abs(colouring[0] - colouring[v - 1])
        for i in range(0, v - 1):
            diff = diff + abs(colouring[i + 1] - colouring[i])
    elif problem_code == "GP":
        diff = abs(partition[0] - partition[v - 1])
        for i in range(0, v - 1):
            diff = diff + abs(partition[i + 1] - partition[i])       
    elif problem_code == "CL":
        diff = abs(clique[0] - clique[v - 1])
        for i in range(0, v - 1):
            diff = diff + abs(clique[i + 1] - clique[i])
    certificate = user_number + alg_number + date_time_number + diff
    f.write("certificate = {0}\n".format(certificate))

    if problem_code == "GC":
        for i in range(0, v):
            f.write("{0},".format(colouring[i]))
            if (i + 1) % 40 == 0:
                f.write("\n")
        if v % 40 != 0:
            f.write("\n")
    if problem_code == "GP":
        for i in range(0, v):
            f.write("{0},".format(partition[i]))
            if (i + 1) % 40 == 0:
                f.write("\n")
        if v % 40 != 0:
            f.write("\n")
    if problem_code == "CL":
        for i in range(0, v):
            f.write("{0},".format(clique[i]))
            if (i + 1) % 40 == 0:
                f.write("\n")
        if v % 40 != 0:
            f.write("\n")

    f.close()
        
    print("witness file 'Witness{0}_{1}.txt' saved for student {2}".format(graph_digit, timestamp, username))

else:

    print("\n*** ERRORS: the witness file has not been saved - fix your errors first!")


















    
