#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the skeleton program 'NegSelTraining.py' around which you should build your implementation.

# The training set should be in a file 'self_training.txt' (in the same folder as this program).

# The output is a detector set that is in the file 'detector_<timestamp>.txt' where '<timestamp>' is a timestamp
# so that you do not overwrite previously produced detector sets. You can always rename these files. However,
# do not tamper with these files in any other way.

# In summary, it is assumed that 'NegSelTraining.py' and 'self_training.txt' are in the same folder
# and that the file containing the detector set is written in this folder.

# As regards the four values to be entered below
# - make sure that no comments are inserted after you have entered the values
# - make sure that 'username' is lower-case
# - make sure that the first two values appear within double quotes
# - make sure that the type of 'threshold' is int or float
# - make sure that the type of 'num_detectors' is int.

# Ensure that your implementation works for data of *general* dimension n and not just for the
# particular dimension of the given data sets!

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "nchw73"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "VD"

#####################################################################################################################
#### ENTER THE THRESHOLD: IF YOU ARE IMPLEMENTING VDETECTOR THEN SET THE THRESHOLD AS YOUR CHOICE OF SELF-RADIUS ####
#####################################################################################################################

threshold = 0.027

######################################################
#### ENTER THE INTENDED SIZE OF YOUR DETECTOR SET ####
######################################################

num_detectors = 600

################################################################
#### DO NOT TOUCH ANYTHING BELOW UNTIL I TELL YOU TO DO SO! ####
####      THIS INCLUDES IMPORTING ADDITIONAL MODULES!       ####
################################################################

import time
import os.path
import random
import math
import sys
    
def get_a_timestamp_for_an_output_file():
    local_time = time.asctime(time.localtime(time.time()))
    timestamp = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
    timestamp = timestamp.replace(" ", "0") 
    return timestamp

def read_points_only(f, point_length, num_points, file):
    list_of_points = []
    count = 0
    error = []
    the_line = f.readline()
    while the_line != "":
        points = the_line.split("[")
        points.pop(0)
        how_many = len(points)
        for i in range(0, how_many):
            if points[i][len(points[i]) - 1] == ",":
                points[i] = points[i][0:len(points[i]) - 2]
            elif points[i][len(points[i]) - 1] == "\n":
                points[i] = points[i][0:len(points[i]) - 3]
            else:
                points[i] = points[i][0:len(points[i]) - 1]
            split_point = points[i].split(",")
            if len(split_point) != point_length:
                error.append("\n*** error: point {0} has the wrong number of components\n".format(i + 1))
                return list_of_points, error
            numeric_point = []
            for j in range(0, point_length):
                numeric_point.append(float(split_point[j]))
            list_of_points.append(numeric_point[:])
            count = count + 1
        the_line = f.readline()
    if count != num_points:
        error.append("\n*** error: there should be {0} points in {1} but there are {2}\n".format(num_points, file, count))
    return list_of_points, error
 
location_of_self = "self_training.txt"

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

#########################################################################################
#### YOU SHOULDN'T HAVE TOUCHED *ANYTHING* UP UNTIL NOW APART FROM SUPPLYING VALUES  ####
#### FOR 'username', 'alg_code', 'threshold' and 'num_detectors' AS REQUESTED ABOVE. ####
####                        NOW READ THE FOLLOWING CAREFULLY!                        ####
#########################################################################################

# The training data has now been read with the following reserved variables:
#   - 'n' = the dimension of the points in the training set                 int
#   - 'threshold' = the threshold or self-radius, as appropriate            int or float
#   - 'Self_num_points' = the number of points in the training set
#   - 'Self' = the list of points in the training set.
# These are reserved variables and their names should not be changed.

# You also have the reserved variables
#   - 'user_name', 'alg_code', 'threshold', 'num_detectors', 'intended_num_detectors' and 'start_time'.
# Remember: if 'alg_code' = 'VD' then 'threshold' denotes your chosen self-radius.

# You need to initialize any other parameters (if you are implementing 'Real-valued Negative Selection'
# or 'VDetector') yourself in your code below.

# The list of detectors that your code generates needs to be stored in the variable 'detectors'.
# This is a reserved variable and has just been initialized as empty above. You need to ensure that
# your computed detector set is stored in 'detectors' as a list of points, i.e., as a list of lists-of-
# floats-of-length-'n' for NS and RV and 'n' + 1 for VD (remember: a detector for VD is a point plus
# its individual radius - see Lecture 4).

# FOR ALL OF THE RESERVED VARIABLES BELOW, YOU MUST ENSURE THAT ON TERMINATION THE TYPE
# OF THE RESPECTIVE VARIABLE IS AS SHOWN.

#  - 'n'                int
#  - 'threshold'        int or float

# Finally, if you choose to use numpy then import it below (and don't forget to ensure that 
# variables are of the correct type on termination).

###########################################
#### NOW YOU CAN ENTER YOUR CODE BELOW ####
###########################################
timed = True
time_limit = 13 # should be 15 in total training and testing

# algorithm: v-detector (S, c0, c1, threshold, intended_num_detectors)

# parameters
c0 = 0.9999 # expected coverage rate for detectors covering non-self
c1 = 0.9999 # expected coverage rate for training set
alpha = 0.36 # ENHANCEMENT - expand detector radius by alpha * threshold

# define euclidean distance between 2 n-dim vectors
def dist(x, y):
    return math.sqrt(sum([(x[i] - y[i])**2 for i in range(n)]))

def v_detector(): # Self is constant in this case, so I've omitted it as parameter
    D = [] # detector set D
    t1 = 0
    start_t = time.time()

    # while there are less than n valid detectors
    while len(D) < intended_num_detectors: 
        t0 = 0
        phase_one_flag = False
        if timed and time.time() - start_t > time_limit:
            return D
        
        while not phase_one_flag:
            # generate a random individual x from [0, 1]^n, set r = inf, and set phase_one_flag = "Successful"
            x = [random.random() for _ in range(n)]
            r = float("inf")
            phase_one_flag = True

            for d in D:
                # if we have collision
                if dist(x, d[:-1]) <= d[-1]:
                    t0 += 1
                    if t0 >= (1 / (1-c0)): # m = 1 / (1-c0)
                        # output D and terminate
                        return D
                    phase_one_flag = False
                    break

        for s in Self: # for every point in training set
            dst = dist(x, s)
            if dst - threshold < r: 
                r = dst - threshold
        
        if r > threshold:
            x.append(r + alpha*threshold)
            D.append(x)
        else:
            t1 += 1

        if t1 >= (1 / (1-c1)):
            # output D and terminate
            return D

    return D

detectors = v_detector()


#########################################################
#### YOU SHOULD HAVE NOW FINISHED ENTERING YOUR CODE ####
####     DO NOT TOUCH ANYTHING BELOW THIS COMMENT    ####
#########################################################

# At this point in the execution, you should have computed
# - the list 'detectors' of your detector set.

now_time = time.time()
training_time = round(now_time - start_time, 1)

timestamp = get_a_timestamp_for_an_output_file()
detector_set_location = "detector_" + timestamp + ".txt"

f = open(detector_set_location, "w")

f.write("username = {0}\n".format(username))
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















    
