########################################
#### PLEASE READ ALL COMMENTS BELOW ####
########################################

# This is the program 'NegSelTesting.py' that allows you to test a detector set that you have constructed
# using your implementation 'NegSelTraining.py' against points from Self and points from non-Self
# - the detector set is read from the detector-set file 'detector.txt'
# - the points from Self and non-Self are in the files 'self_testing.txt' and 'non_self_testing.txt', respectively.
#
# The file 'detector.txt' must be in the format as supplied as output from 'NegSelTraining.py'.
# It is assumed that 'NegSelTesting.py', 'detector.txt', 'self_testing.txt' and 'non_self_testing.txt' are in the same folder.

##############################################################
#### YOU SHOULD NOT AMEND ANYTHING IN THIS PROGRAM AT ALL ####
##############################################################

import sys
import os.path
import math
import time

def get_location_of_self_testing(self_testing):
    self_location = self_testing
    return self_location

def get_location_of_non_self_testing(non_self_testing):
    non_self_location = non_self_testing
    return non_self_location

def get_location_of_detector_set(detector_set):
    detector_set_location = detector_set
    return detector_set_location

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

def Euclidean_distance(n, first_individual, second_individual):
    distance = 0
    for i in range(0, n):
        distance = distance + (first_individual[i] - second_individual[i])**2
    distance = math.sqrt(distance)
    return distance

def testing(n, alg, detector_set, num_detectors, threshold, individual):
    detection = False
    for i in range(0, num_detectors):
        if alg == "NS" or alg == "RV":
            distance = Euclidean_distance(n, individual, detector_set[i])
            if distance <= threshold:
                detection = True
                break
        else:
            truncated_detector = detector_set[i][0:n - 1]
            distance = Euclidean_distance(n - 1, individual, truncated_detector)
            if distance <= detector_set[i][n - 1]:
                detection = True
                break
    return detection

if __name__ == "__main__":

    detector_set = "detector.txt"
    detector_set_location = get_location_of_detector_set(detector_set)
    self_testing = "self_testing.txt"
    self_location = get_location_of_self_testing(self_testing)
    non_self_testing = "non_self_testing.txt"
    non_self_location = get_location_of_non_self_testing(non_self_testing)

    if not os.path.exists(detector_set_location):
        print("\n*** error: {0} does not exist\n".format(detector_set_location))
        sys.exit()
    elif not os.path.exists(self_location):
        print("\n*** error: {0} does not exist\n".format(self_location))
        sys.exit()
    elif not os.path.exists(non_self_location):
        print("\n*** error: {0} does not exist\n".format(non_self_location))
        sys.exit()

    f = open(detector_set_location, "r")

    username = f.readline()
    length_of_username = len(username)
    username = username[len("username = "):length_of_username - 1]
    detector = f.readline()
    if detector != "detector set\n":
        print("\n*** error: the file {0} is not denoted as a detector set\n".format(detector_set_location))
        f.close()
        sys.exit()
    alg_code = f.readline()
    length_of_alg_code = len(alg_code)
    alg_code = alg_code[len("algorithm code = "):length_of_alg_code - 1]
    if not alg_code in ["NS", "RV", "VD"]:
        print("\n*** error: the algorithm code {0} from detector.txt is illegal\n".format(alg_code))
        f.close()
        sys.exit()
    dim = f.readline()
    length_of_dim = len(dim)
    dim = dim[len("dimension = "):length_of_dim - 1]
    if alg_code != "VD":
        detector_point_length = int(dim)
    else:
        detector_point_length = int(dim) + 1
    if alg_code != "VD":
        threshold = f.readline()
        length_of_threshold = len(threshold)
        threshold = threshold[len("threshold = "):length_of_threshold - 1]
        threshold = float(threshold)
    else:
        threshold = f.readline()
        length_of_threshold = len(threshold)
        threshold = threshold[len("self-radius = "):length_of_threshold - 1]
        threshold = float(threshold)
    num_detectors = f.readline()
    suffix = " (from an intended number of "
    start_suffix = num_detectors.find(suffix)
    actual_num_detectors = num_detectors[len("number of detectors = "):start_suffix]
    actual_num_detectors = int(actual_num_detectors)
    original_num_detectors = num_detectors[start_suffix + len(suffix):len(num_detectors) - 2]
    original_num_detectors = int(original_num_detectors)
    training_time = f.readline()
    length_of_training_time = len(training_time)
    training_time = training_time[len("training time = "):length_of_training_time - 1]
    training_time = round(float(training_time), 3)

    list_of_detectors, error = read_points_only(f, detector_point_length, actual_num_detectors, detector_set_location)
    detectors = list_of_detectors[:]

    f.close()

    if error != []:
        length = len(error)
        for i in range(0, length):
            print(error[i])
        sys.exit()

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
    Self = list_of_points[:]

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

    print("\nevaluating detector set ...\n")

    start_time = time.time()
    
    FP = 0
    TN = 0
    TP = 0
    FN = 0
    for i in range(0, self_num_points):
        detection = testing(detector_point_length, alg_code, detectors, actual_num_detectors, threshold, Self[i])
        if detection == True:
            FP = FP + 1
        else:
            TN = TN + 1
    for i in range(0, non_self_num_points):
        detection = testing(detector_point_length, alg_code, detectors, actual_num_detectors, threshold, non_Self[i])
        if detection == True:
            TP = TP + 1
        else:
            FN = FN + 1

    now_time = time.time()
    testing_time = round(now_time - start_time, 1)

    print("This detector set was built by '{0}' using algorithm '{1}' on data of dimension {2}.".format(username, alg_code, self_point_length))
    if alg_code != "VD":
        print("The threshold distance is {0} and the time to build was {1}.".format(threshold, training_time))
    else:
        print("The self-radius is {0} and the time to build was {1}.".format(threshold, training_time))
    print("From {0} test-individuals from Self and {1} test-individuals from non-Self:".format(self_num_points, non_self_num_points))
    print("   - detection rate   TP/(TP+FN) = {0}%".format(round(100 * TP / (TP + FN), 2)))
    print("   - false alarm rate FP/(FP+TN) = {0}%".format(round(100 * FP / (FP + TN), 2)))
    print("   - elapsed testing time        = {0}".format(testing_time))
