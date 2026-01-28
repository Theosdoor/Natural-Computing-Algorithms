import time
import os.path
import random
import math
import sys
import numpy
from utils import get_a_timestamp_for_an_output_file, read_points_only, Euclidean_distance, euclidean_distance

alg_code = "VD"
threshold = 0.027
num_detectors = 1000

# Get the project root directory (parent of nchw73/)
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

# --------from TESTING ----------------
def get_location_of_self_testing(self_testing):
    self_location = self_testing
    return self_location

def get_location_of_non_self_testing(non_self_testing):
    non_self_location = non_self_testing
    return non_self_location

def testing(n, alg, detector_set, num_detectors, threshold, individual):
    detection = False
    for i in range(0, num_detectors):
        try:
            truncated_detector = detector_set[i][0:n - 1]
        except:
            print(detector_set[i])
            sys.exit()
        distance = Euclidean_distance(n - 1, individual, truncated_detector)
        if distance <= detector_set[i][n - 1]:
            detection = True
            break
    return detection

# get test sets
self_testing = os.path.join(project_root, "data", "self_testing.txt")
self_location = get_location_of_self_testing(self_testing)
non_self_testing = os.path.join(project_root, "data", "non_self_testing.txt")
non_self_location = get_location_of_non_self_testing(non_self_testing)

if not os.path.exists(self_location):
    print("\n*** error: {0} does not exist\n".format(self_location))
    sys.exit()
elif not os.path.exists(non_self_location):
    print("\n*** error: {0} does not exist\n".format(non_self_location))
    sys.exit()

detector_point_length = int(dim) + 1
actual_num_detectors = num_detectors
original_num_detectors = intended_num_detectors

f = open(self_location, "r")

self_or_non_self = f.readline()
if self_or_non_self != "Self\n":
    print("\n*** error: the file {0} is not denoted as a Self-file\n".format(self_location))
    f.close()
    sys.exit()
dim = f.readline()
length_of_dim = len(dim)
dim = dim[len("n = "):length_of_dim - 1]
self_point_length = int(dim)
num_points = f.readline()
length_of_num_points = len(num_points)
num_points = num_points[len("number of points = "):length_of_num_points - 1]
self_num_points = int(num_points)

list_of_points, error = read_points_only(f, self_point_length, self_num_points, self_location)
Self_test = list_of_points[:]

f.close()

if error != []:
    length = len(error)
    for i in range(0, length):
        print(error[i])
    sys.exit()

f = open(non_self_location, "r")

self_or_non_self = f.readline()
if self_or_non_self != "non-Self\n":
    print("\n*** error: the file {0} is not denoted as a non-Self-file\n".format(non_self_location))
    f.close()
    sys.exit()
dim = f.readline()
length_of_dim = len(dim)
dim = dim[len("n = "):length_of_dim - 1]
non_self_point_length = int(dim)
num_points = f.readline()
length_of_num_points = len(num_points)
num_points = num_points[len("number of points = "):length_of_num_points - 1]
non_self_num_points = int(num_points)

list_of_points, error = read_points_only(f, non_self_point_length, non_self_num_points, non_self_location)
non_Self = list_of_points[:]

f.close()

if error != []:
    length = len(error)
    for i in range(0, length):
        print(error[i])
    sys.exit()

# test fn
def test(detectors):
    start_time = time.time()
    
    FP = 0
    TN = 0
    TP = 0
    FN = 0
    for i in range(0, self_num_points):
        detection = testing(detector_point_length, alg_code, detectors, len(detectors), threshold, Self_test[i])
        if detection == True:
            FP = FP + 1
        else:
            TN = TN + 1
    for i in range(0, non_self_num_points):
        detection = testing(detector_point_length, alg_code, detectors, len(detectors), threshold, non_Self[i])
        if detection == True:
            TP = TP + 1
        else:
            FN = FN + 1

    now_time = time.time()
    testing_time = round(now_time - start_time, 1)
    det_rate = round(100 * TP / (TP + FN), 2)
    far = round(100 * FP / (FP + TN), 2)

    return det_rate, far, testing_time

added_note = ""
time_limit = 13 # should be 15 in total training and testing

# algorithm: v-detector (S, c0, c1, threshold, intended_num_detectors)

# parameters
# alpha = c0/c1
# c0 = 0.9999 # expected coverage rate for detectors covering non-self
# c1 = 0.9999 # expected coverage rate for training set

p = 0.5  # detector coverage
sample_size = max(5/p, 5/(1-p))

def v_detector(c0, c1, r_self, intended_num_detectors, alpha):
    D = [] # detector set D
    t1 = 0
    start_t = time.time()

    # while there are less than n valid detectors
    while len(D) < intended_num_detectors: 
        t0 = 0
        phase_one_flag = False
        if time.time() - start_t > time_limit:
            return D
        
        while not phase_one_flag:
            # generate a random individual x from [0, 1]^n, set r = inf, and set phase_one_flag = "Successful"
            x = [random.random() for _ in range(n)]
            r = float("inf")
            # TODO generate randomly? or near to another detector?
            phase_one_flag = True

            for d in D:
                # if we have collision
                if euclidean_distance(x, d[:-1]) <= d[-1]:
                    t0 += 1
                    if t0 >= (1 / (1-c0)): # m = 1 / (1-c0)
                        # output D and terminate
                        return D
                    phase_one_flag = False
                    break

        for s in Self:  # for every point in training set
            dst = euclidean_distance(x, s)
            if dst - r_self < r: 
                r = dst - r_self
        
        if r > r_self:
            x.append(r + alpha*threshold) # ENHANCEMENT?
            D.append(x)
        else:
            t1 += 1

        if t1 >= (1 / (1-c1)):
            # output D and terminate
            return D

    return D



# ----------------- parameter tuning -----------------
# detectors = v_detector(Self, c0, c1, threshold, intended_num_detectors)
# det_rate, far, testing_time = test(detectors)

# grid search to get params that minimize far and maximize det_rate
# params to tune: c0, c1, threshold, intended_num_detectors
goodEnough = False
sat_far = 10 # break if v and far less than this
sat_det = 85 # break if ^ and det rate higher than this


max_far = 6
min_det = 80
last_best_update = -1

best_c0 = 0.9999
best_c1 = 0.9999
best_threshold = threshold
best_intended_num_detectors = intended_num_detectors

best_det_rate = 0
best_far = 100
best_detectors = []

# uniformly dist values to iterate over
Ns = [600]#numpy.linspace(400, 1000, 7)
# c0s = numpy.linspace(0.99, 1, 10, endpoint=False)
c0s = [0.9999]
c1s = [0.9999]
# c1s = numpy.linspace(0.9999, 1, 10, endpoint=False)
thresholds = [0.027]#numpy.linspace(0.0255, 0.0275, 11)
boundaries = numpy.linspace(0.35, 0.44, 10) # good between 0.35-0.5

max_it = 3

# N = 1000
# c0 = 0.9999
# c1 = 0.9999
# threshold = 0.0287

for N in Ns:
    if goodEnough:
        break
    for c0 in c0s:
        if goodEnough:
            break
        # c0 = round(c0, 4)
        for c1 in c1s:
            if goodEnough:
                break
            # c1 = round(c1, 4)
            for threshold in thresholds:
                threshold = round(threshold, 4)
                for alpha in boundaries:
                    alpha = round(alpha, 4)
                    print("***********")
                    print("Testing: c0: {0}, c1: {1}, threshold: {2}, N: {3}, alpha: {4}".format(c0, c1, threshold, N, alpha))
                    if goodEnough:
                        break
                    # it = 0
                    # while it < max_it: # repeat each parameter combo x times to account for randomness
                    avg_det_rate = 0
                    avg_far = 0
                    avd_len = 0
                    for it in range(max_it):
                        # print('-----------------')
                        # print("Iteration: {0}".format(it))
                        detectors = v_detector(c0, c1, threshold, N, alpha)
                        det_rate, far, testing_time = test(detectors)
                        avg_det_rate += det_rate
                        avg_far += far
                        avd_len += len(detectors)
                    
                    det_rate = round(avg_det_rate / max_it, 2)
                    far = round(avg_far / max_it, 2)
                    avg_len = avd_len // max_it
                    print("det_rate: {0}, far: {1}".format(det_rate, far))
                    print("Num detectors: {0}".format(avg_len))

                    if det_rate > sat_det and far < sat_far:
                        goodEnough = True
                        break

detectors = best_detectors

sys.exit()

now_time = time.time()
training_time = round(now_time - start_time, 1)

timestamp = get_a_timestamp_for_an_output_file()
detector_set_location = os.path.join(project_root, "outputs", "detector_TT_" + timestamp + ".txt")

f = open(detector_set_location, "w")

f.write("detector set\n")
f.write("algorithm code = {0}\n".format(alg_code))
f.write("dimension = {0}\n".format(n))
if alg_code != "VD":
    f.write("threshold = {0}\n".format(threshold))
else:
    f.write("self-radius = {0}\n".format(threshold))
num_detectors = len(detectors)
f.write("number of detectors = {0} (from an intended number of {1})\n".format(num_detectors, intended_num_detectors))
f.write("training time = {0}\n".format(training_time))
f.write(added_note)
detector_length = n
if alg_code == "VD":
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



# TESTING
print("This detector set was built using algorithm '{0}' on data of dimension {1}.".format(alg_code, self_point_length))
print("The self-radius is {0} and the time to build was {1}.".format(threshold, training_time))
print("From {0} test-individuals from Self and {1} test-individuals from non-Self:".format(self_num_points, non_self_num_points))
print("   - detection rate   TP/(TP+FN) = {0}%".format(det_rate))
print("   - false alarm rate FP/(FP+TN) = {0}%".format(far))
print("   - elapsed testing time        = {0}".format(testing_time))












    
