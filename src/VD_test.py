import sys
import os.path
import math
import time
from tqdm import tqdm
from utils import read_points_only, Euclidean_distance

# Get the project root directory (parent of src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def get_location_of_self_testing(self_testing):
    return os.path.join(project_root, "data", self_testing)

def get_location_of_non_self_testing(non_self_testing):
    return os.path.join(project_root, "data", non_self_testing)

def get_location_of_detector_set(detector_set):
    # If full path provided, use it; otherwise assume it's in outputs/
    if os.path.isabs(detector_set) or os.path.exists(detector_set):
        return detector_set
    return os.path.join(project_root, "outputs", detector_set)

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

    # Accept detector set filename as command line argument
    if len(sys.argv) > 1:
        detector_set = sys.argv[1]
    else:
        # Default to most recent detector file in outputs/
        outputs_dir = os.path.join(project_root, "outputs")
        detector_files = [f for f in os.listdir(outputs_dir) if f.startswith("detector_") and f.endswith(".txt")]
        
        if not detector_files:
            print("\n*** error: No detector files found in outputs/")
            print("Please run VD_train.py first or specify a detector file as argument")
            print("Usage: python VD_test.py <detector_file.txt>")
            sys.exit()
        
        # Sort by modification time, most recent first
        detector_files.sort(key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)), reverse=True)
        detector_set = detector_files[0]
        print(f"Using most recent detector file: {detector_set}")
    
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
    
    print("Testing Self data...")
    for i in tqdm(range(0, self_num_points), desc="Testing Self", unit="point"):
        detection = testing(detector_point_length, alg_code, detectors, actual_num_detectors, threshold, Self[i])
        if detection == True:
            FP = FP + 1
        else:
            TN = TN + 1
    
    print("Testing non-Self data...")
    for i in tqdm(range(0, non_self_num_points), desc="Testing non-Self", unit="point"):
        detection = testing(detector_point_length, alg_code, detectors, actual_num_detectors, threshold, non_Self[i])
        if detection == True:
            TP = TP + 1
        else:
            FN = FN + 1

    now_time = time.time()
    testing_time = round(now_time - start_time, 1)

    print("This detector set was built using algorithm '{0}' on data of dimension {1}.".format(alg_code, self_point_length))
    if alg_code != "VD":
        print("The threshold distance is {0} and the time to build was {1}.".format(threshold, training_time))
    else:
        print("The self-radius is {0} and the time to build was {1}.".format(threshold, training_time))
    print("From {0} test-individuals from Self and {1} test-individuals from non-Self:".format(self_num_points, non_self_num_points))
    print("   - detection rate   TP/(TP+FN) = {0}%".format(round(100 * TP / (TP + FN), 2)))
    print("   - false alarm rate FP/(FP+TN) = {0}%".format(round(100 * FP / (FP + TN), 2)))
    print("   - elapsed testing time        = {0}".format(testing_time))
