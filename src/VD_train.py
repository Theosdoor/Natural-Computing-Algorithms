threshold = 0.027

num_detectors = 600

import time
import os.path
import random
import math
import sys
from tqdm import tqdm
from utils import get_a_timestamp_for_an_output_file, read_points_only, euclidean_distance
 
# Get the project root directory (parent of src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
location_of_self = os.path.join(project_root, "data", "self_training.txt")

if not os.path.exists(location_of_self):
    print("\n*** error: {0} does not exist\n".format(location_of_self))
    sys.exit()

f = open(location_of_self, "r")

self_or_non_self = f.readline()
if self_or_non_self != "Self\n":
    print("\n*** error: the file " + location_of_self + " is not denoted as a Self-file\n")
    f.close()
    sys.exit()
dim = f.readline()
length_of_dim = len(dim)
dim = dim[len("n = "):length_of_dim - 1]
n = int(dim)
num_points = f.readline()
length_of_num_points = len(num_points)
num_points = num_points[len("number of points = "):length_of_num_points - 1]
Self_num_points = int(num_points)

list_of_points, error = read_points_only(f, n, Self_num_points, location_of_self)
Self = list_of_points[:]

f.close()

if error != []:
    length = len(error)
    for i in range(0, length):
        print(error[i])
    sys.exit()

detectors = []

start_time = time.time()

intended_num_detectors = num_detectors

# VDetector algorithm parameters
timed = True
time_limit = 13  # should be 15 in total training and testing

# algorithm: v-detector (S, c0, c1, threshold, intended_num_detectors)

c0 = 0.9999  # expected coverage rate for detectors covering non-self
c1 = 0.9999  # expected coverage rate for training set
alpha = 0.36  # ENHANCEMENT - expand detector radius by alpha * threshold

def v_detector(): # Self is constant in this case, so I've omitted it as parameter
    D = [] # detector set D
    t1 = 0
    start_t = time.time()

    # while there are less than n valid detectors
    with tqdm(total=intended_num_detectors, desc="Generating detectors", unit="detector") as pbar:
        while len(D) < intended_num_detectors: 
            t0 = 0
            phase_one_flag = False
            if timed and time.time() - start_t > time_limit:
                pbar.write(f"Time limit reached. Generated {len(D)}/{intended_num_detectors} detectors")
                return D
            
            while not phase_one_flag:
                # generate a random individual x from [0, 1]^n, set r = inf, and set phase_one_flag = "Successful"
                x = [random.random() for _ in range(n)]
                r = float("inf")
                phase_one_flag = True

                for d in D:
                    # if we have collision
                    if euclidean_distance(x, d[:-1]) <= d[-1]:
                        t0 += 1
                        if t0 >= (1 / (1-c0)): # m = 1 / (1-c0)
                            # output D and terminate
                            pbar.write(f"Coverage threshold reached. Generated {len(D)}/{intended_num_detectors} detectors")
                            return D
                        phase_one_flag = False
                        break

            for s in Self:  # for every point in training set
                dst = euclidean_distance(x, s)
                if dst - threshold < r: 
                    r = dst - threshold
            
            if r > threshold:
                x.append(r + alpha*threshold)
                D.append(x)
                pbar.update(1)
                pbar.set_postfix({'time_left': f"{time_limit - (time.time() - start_t):.1f}s"})
            else:
                t1 += 1

            if t1 >= (1 / (1-c1)):
                # output D and terminate
                pbar.write(f"Training coverage threshold reached. Generated {len(D)}/{intended_num_detectors} detectors")
                return D

    return D

detectors = v_detector()

now_time = time.time()
training_time = round(now_time - start_time, 1)

timestamp = get_a_timestamp_for_an_output_file()
detector_set_location = os.path.join(project_root, "outputs", "detector_" + timestamp + ".txt")

f = open(detector_set_location, "w")

f.write("detector set\n")
f.write("algorithm code = VD\n")
f.write("dimension = {0}\n".format(n))
f.write("self-radius = {0}\n".format(threshold))
num_detectors = len(detectors)
f.write("number of detectors = {0} (from an intended number of {1})\n".format(num_detectors, intended_num_detectors))
f.write("training time = {0}\n".format(training_time))
detector_length = n + 1
for i in range(0, num_detectors):
    f.write("[")
    for j in range(0, detector_length):
        if j != detector_length - 1:
            f.write("{0},".format(detectors[i][j]))
        else:
            f.write("{0}]".format(detectors[i][j]))
            if i == num_detectors - 1:
                f.write("\n")
            else:
                f.write(",\n")
f.close()

print("detector set saved as {0}\n".format(detector_set_location))















    
