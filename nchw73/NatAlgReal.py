#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the skeleton program 'NatAlgReal.py' around which you should build your implementation.
# Please read through this program and follow the instructions given.

# There are no input or output files, with the results printed to the standard output.

# As regards the two values to be entered below
# - make sure that the first two values appear within double quotes
# - make sure that 'username' is lower-case
# - make sure that no comments are inserted after you have entered the values.

# Ensure that your implementation works for *arbitrary* hard-coded functions of arbitrary
# dimension and arbitrary min- and max-ranges!

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "nchw73"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "CS"

################################################################
#### DO NOT TOUCH ANYTHING BELOW UNTIL I TELL YOU TO DO SO! ####
####      THIS INCLUDES IMPORTING ADDITIONAL MODULES!       ####
################################################################

import time
import random
import math
import sys
import os
import datetime

def compute_f(point):
    f = -1 * math.sin(point[0])*math.sqrt(point[0]) * math.sin(point[1])*math.sqrt(point[1]) * \
        math.sin(point[2])*math.sqrt(point[2]) * math.sin(point[3])*math.sqrt(point[3])
    return f

n = 4

min_range = [0, 0, 0, 0]
max_range = [10, 10, 10, 10]

start_time = time.time()

#########################################################################################
#### YOU SHOULDN'T HAVE TOUCHED *ANYTHING* UP UNTIL NOW APART FROM SUPPLYING VALUES  ####
####                 FOR 'username' and 'alg_code' AS REQUESTED ABOVE.               ####
####                        NOW READ THE FOLLOWING CAREFULLY!                        ####
#########################################################################################

# The function 'f' is 'n'-dimensional and you are attempting to MINIMIZE it.
# To compute the value of 'f' at some point 'point', where 'point' is a list of 'n' integers or floats,
# call the function 'compute_f(point)'.
# The ranges for the values of the components of 'point' are given above. The lists 'min_range' and
# 'max_range' above hold the minimum and maximum values for each component and you should use these
# list variables in your code.

# On termination your algorithm should be such that:
#   - the reserved variable 'min_f' holds the minimum value that you have computed for the
#     function 'f' 
#   - the reserved variable 'minimum' is a list of 'n' entries (integer or float) holding the point at which
#     your value of 'min_f' is attained.

# Note that the variables 'username', 'alg_code', 'f', 'point', 'min_f', 'n', 'min_range', 'max_range' and
# 'minimum' are all reserved.

# FOR THE RESERVED VARIABLES BELOW, YOU MUST ENSURE THAT ON TERMINATION THE TYPE
# OF THE RESPECTIVE VARIABLE IS AS SHOWN.

#  - 'min_f'                int or float
#  - 'minimum'              list of int or float

# You should ensure that your code works on any function hard-coded as above, using the
# same reserved variables and possibly of a dimension different to that given above. I will
# run your code with a different such function/dimension to check that this is the case.

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
#   - 'f_min' = minimum frequency                       float or int
#   - 'f_max' = maximum frequency                       float or int

# These are reserved variables and need to be treated as such, i.e., use these names for these
# parameters and don't re-use the names. Don't forget to ensure that on termination all the above
# variables have the stated type. In particular, if you use specific numpy types then you'll need
# to ensure that they are changed prior to termination (this is checked).

# INITIALIZE THE ACTUAL PARAMETERS YOU USE FOR YOUR ALGORITHM BELOW. ENSURE THAT YOU INITIALIZE
# *ALL* OF THE PARAMETERS REQUIRED APPROPRIATELY (SEE ABOVE) FOR YOUR CHOSEN ALGORITHM.

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

num_cyc = 24000
N = 50 # number of nests
p = 0.6 # fraction of local flights to undertake
q = 0.25 # fraction of nests to abandon
alpha = 0.4*n # scaling factor for Levy flights

timed = True
max_time = 9.9 # maximum time in seconds
start_t = time.time()

min_f = 0 # keep track of minimum f value found throughout

# for Mantegna
beta = 1.5
sigma = ((math.gamma(1 + beta) * math.sin(math.pi * beta / 2)) / (beta * math.gamma((1 + beta) / 2) * 2 ** ((beta - 1) / 2))) ** (1 / beta)

###########################################
#### NOW INCLUDE THE REST OF YOUR CODE ####
###########################################

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
    # randomly generate population of nests
    P = []
    for i in range(N):
        P.append([random.uniform(min_range[j], max_range[j]) for j in range(n)])

    # compute fitness of each nest
    fitnesses = [fitness(P[i]) for i in range(N)]
    min_f = min(fitnesses)
    minima = [P[fitnesses.index(min_f)]] # list of minima

    # main loop
    for t in range(num_cyc):
        # alpha = alpha / (10*t+1)**0.5 # decrease alpha over time

        if timed and time.time() - start_t > max_time:
            print("Time limit reached")
            return min_f, minima
        
        if t % 2000 == 0:
            print("Cycle {0}, min_f: {1}".format(t, min_f))
            print(minima)

        # levy flights from each nest
        for i in range(N):
            # undertake Levy flight from x_i to y_i
            y = levy_flight(P[i], alpha) # y_i
            if fitness(y) < fitnesses[i]:
                # replace x_i with y_i if y_i is better
                P[i] = y
                fitnesses[i] = fitness(y)

        # local search with probability p
        local_flights = random.sample(range(N), int(p * N))
        for j in local_flights:
            # undertake local flight
            y = local_flight(P[j])
            if fitness(y) < fitnesses[j]:
                P[j] = y
                fitnesses[j] = fitness(y)

            if timed and time.time() - start_t > max_time:
                print("Time limit reached")
                return min_f, minima

        # rank nests by fitness
        sorted_indices = sorted(range(N), key=lambda x: fitnesses[x])

        # update min_f and minima if necessary
        if fitnesses[sorted_indices[0]] < min_f:
            min_f = fitnesses[sorted_indices[0]]
            for i in range(1, len(sorted_indices)):
                if fitnesses[sorted_indices[i]] == min_f:
                    minima.append(P[sorted_indices[i]])
                else:
                    break

        # abandon q fraction of worst nests
        abandoned_nests = sorted_indices[-int(q * N):]
        for k in abandoned_nests:
            # generate new random nest to replace TODO maybe not random?
            P[k] = [random.uniform(min_range[j], max_range[j]) for j in range(n)]
            P[k] = levy_flight(P[k], alpha) # levy flight from new nest
            fitnesses[k] = fitness(P[k])

            # update min_f if necessary
            if fitnesses[k] < min_f:
                min_f = fitnesses[k]
                minima = [P[k]] # reset minima
            elif fitnesses[k] == min_f:
                minima.append(P[k])
            
            if timed and time.time() - start_t > max_time:
                print("Time limit reached")
                return min_f, minima
        
    return min_f, minima

x = cuckoo_search(N, num_cyc, p, q, alpha)
try:
    min_f, minima = x
    minimum = minima[0]
except:
    print(x)
    sys.exit()

print(min_f-100)

# try genetic algorithm with parameters for cuckoo search
# tune parameters: N, p, q, alpha


# population of parameter combos
# P_size = 100
# P = []
# for i in range(P_size):
#     test_N = np.random.randint(10, 100)
#     test_p = np.random.uniform(0.1, 1)
#     test_q = np.random.uniform(0.1, 1)
#     test_alpha = np.random.uniform(0.5, 2)
#     test = [test_N, test_p, test_q, test_alpha]

#     while test in P:
#         # ensure no duplicates
#         test_N = np.random.randint(10, 100)
#         test_p = np.random.uniform(0.1, 1)
#         test_q = np.random.uniform(0.1, 1)
#         test_alpha = np.random.uniform(0.5, 2)
#         test = [test_N, test_p, test_q, test_alpha]

#     P.append(test)






#########################################################
#### YOU SHOULD HAVE NOW FINISHED ENTERING YOUR CODE ####
####     DO NOT TOUCH ANYTHING BELOW THIS COMMENT    ####
#########################################################

# At this point in the execution, you should have computed your minimum value for the function 'f' in the
# variable 'min_f' and the variable 'minimum' should hold a list containing the values of the point 'point'
# for which function 'f(point)' attains your minimum.

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)

error = []

try:
    n
    try:
        y = n
    except:
        error.append("*** error: 'n' has not been initialized")
        n = -1
except:
    error.append("*** error: the variable 'n' does not exist\n")
    n = -1
try:
    num_cyc
    try:
        y = num_cyc
    except:
        error.append("*** error: 'num_cyc' has not been initialized")
        num_cyc = -1
except:
    error.append("*** error: the variable 'num_cyc' does not exist")
    num_cyc = -1

if alg_code == "AB":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        M
        try:
           y = M
        except:
            error.append("*** error: 'M' has not been initialized")
            M = -1
    except:
        error.append("*** error: the variable 'M' does not exist")
        M = -1
    try:
        lambbda
        try:
           y = lambbda
        except:
            error.append("*** error: 'lambbda' has not been initialized")
            lambbda = -1
    except:
        error.append("*** error: the variable 'lambbda' does not exist")
        lambbda = -1
if alg_code == "FF":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        alpha
        try:
           y = alpha
        except:
            error.append("*** error: 'alpha' has not been initialized")
            alpha = -1
    except:
        error.append("*** error: the variable 'alpha' does not exist")
        alpha = -1
    try:
        lambbda
        try:
           y = lambbda
        except:
            error.append("*** error: 'lambbda' has not been initialized")
            lambbda = -1
    except:
        error.append("*** error: the variable 'lambbda' does not exist")
        lambbda = -1
if alg_code == "CS":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        p
        try:
           y = p
        except:
            error.append("*** error: 'p' has not been initialized")
            p = -1
    except:
        error.append("*** error: the variable 'p' does not exist")
        p = -1
    try:
        q
        try:
           y = q
        except:
            error.append("*** error: 'q' has not been initialized")
            q = -1
    except:
        error.append("*** error: the variable 'q' does not exist")
        q = -1
    try:
        alpha
        try:
           y = alpha
        except:
            error.append("*** error: 'alpha' has not been initialized")
            alpha = -1
    except:
        error.append("*** error: the variable 'alpha' does not exist")
        alpha = -1
    try:
        beta
        try:
           y = beta
        except:
            error.append("*** error: 'beta' has not been initialized")
            beta = -1
    except:
        error.append("*** error: the variable 'beta' does not exist")
        beta = -1
if alg_code == "WO":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        b
        try:
           y = b
        except:
            error.append("*** error: 'b' has not been initialized")
            b = -1
    except:
        error.append("*** error: the variable 'b' does not exist")
        b = -1
if alg_code == "BA":
    try:
        sigma
        try:
           y = sigma
        except:
            error.append("*** error: 'sigma' has not been initialized")
            sigma = -1
    except:
        error.append("*** error: the variable 'sigma' does not exist")
        sigma = -1
    try:
        f_max
        try:
           y = f_max
        except:
            error.append("*** error: the variable 'f_max' has not been initialized")
            f_max = -1
    except:
        error.append("*** error: the variable 'f_max' does not exist")
        f_max = -1
    try:
        f_min
        try:
           y = f_min
        except:
            error.append("*** error: 'f_min' has not been initialized")
            f_min = -1
    except:
        error.append("*** error: the variable 'f_min' does not exist")
        f_min = -1

if type(n) != int:
    error.append("*** error: 'n' is not an integer: it is {0} and it has type {1}".format(n, type(n)))
if type(num_cyc) != int:
    error.append("*** error: 'num_cyc' is not an integer: it is {0} and it has type {1}".format(num_cyc, type(num_cyc)))

if alg_code == "AB":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(M) != int:
        error.append("*** error: 'M' is not an integer: it is {0} and it has type {1}".format(M, type(M)))
    if type(lambbda) != int and type(lambbda) != float:
        error.append("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1}".format(lambbda, type(lambbda)))

if alg_code == "FF":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(lambbda) != int and type(lambbda) != float:
        error.append("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1}".format(lambbda, type(lambbda)))
    if type(alpha) != int and type(alpha) != float:
        error.append("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1}".format(alpha, type(alpha)))

if alg_code == "CS":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(p) != int and type(p) != float:
        error.append("*** error: 'p' is not an integer or a float: it is {0} and it has type {1}".format(p, type(p)))
    if type(q) != int and type(q) != float:
        error.append("*** error: 'q' is not an integer or a float: it is {0} and it has type {1}".format(q, type(q)))
    if type(alpha) != int and type(alpha) != float:
        error.append("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1}".format(alpha, type(alpha)))
    if type(beta) != int and type(beta) != float:
        error.append("*** error: 'beta' is not an integer or a float: it is {0} and it has type {1}".format(beta, type(beta)))

if alg_code == "WO":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}\n".format(N, type(N)))
    if type(b) != int and type(b) != float:
        error.append("*** error: 'b' is not an integer or a float: it is {0} and it has type {1}".format(b, type(b)))

if alg_code == "BA":
    if type(sigma) != int and type(sigma) != float:
        error.append("*** error: 'sigma' is not an integer or a float: it is {0} and it has type {1}".format(sigma, type(sigma)))
    if type(f_min) != int and type(f_min) != float:
        error.append("*** error: 'f_min' is not an integer or a float: it is {0} and it has type {1}".format(f_min, type(f_min)))
    if type(f_max) != int and type(f_max) != float:
        error.append("*** error: 'f_max' is not an integer or a float: it is {0} and it has type {1}".format(f_max, type(f_max)))

if type(min_f) != int and type(min_f) != float:
    error.append("*** error: there is no real-valued variable 'min_f'")
if type(minimum) != list:
    error.append("*** error: there is no tuple 'minimum' giving the minimum point")
elif type(n) == int and len(minimum) != n:
    error.append("*** error: there is no {0}-tuple 'minimum' giving the minimum point; you have a {1}-tuple".format(n, len(minimum)))
elif type(n) == int:
    for i in range(0, n):
        if not "int" in str(type(minimum[i])) and not "float" in str(type(minimum[i])):
            error.append("*** error: the value for component {0} (ranging from 1 to {1}) in the minimum point is not numeric\n".format(i + 1, n))

if error != []:
    print("\n*** ERRORS: there were errors in your execution:")
    length = len(error)
    for i in range(0, length):
        print(error[i])
    print("\n Fix these errors and run your code again.\n")
else:
    print("\nYou have found a minimum value of {0} and a minimum point of {1}.".format(min_f, minimum))
    print("Your elapsed time was {0} seconds.\n".format(elapsed_time))
    
