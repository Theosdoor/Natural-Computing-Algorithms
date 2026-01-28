"""
Utility functions for Natural Computing Algorithms coursework.

This module consolidates common functions used across multiple algorithm implementations.
"""

import time
import os
import math


def get_a_timestamp_for_an_output_file():
    """
    Generate a timestamp string for output filenames.
    
    Returns:
        str: Timestamp in format 'MmmDDHHMMSS' (e.g., 'Jan2814:3045')
    """
    local_time = time.asctime(time.localtime(time.time()))
    timestamp = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
    timestamp = timestamp.replace(" ", "0") 
    return timestamp


def euclidean_distance(x, y, n=None):
    """
    Calculate Euclidean distance between two n-dimensional points.
    
    Args:
        x: First point (list of floats)
        y: Second point (list of floats)
        n: Dimension (optional, inferred from x if not provided)
    
    Returns:
        float: Euclidean distance
    """
    if n is None:
        n = len(x)
    distance = 0
    for i in range(n):
        distance = distance + (x[i] - y[i])**2
    return math.sqrt(distance)


# Legacy function name for backwards compatibility
def Euclidean_distance(n, first_individual, second_individual):
    """Legacy wrapper for euclidean_distance with original signature."""
    return euclidean_distance(first_individual, second_individual, n)


def read_points_only(f, point_length, num_points, file):
    """
    Read point data from an open file object.
    
    Args:
        f: Open file object
        point_length: Expected dimension of each point
        num_points: Expected number of points
        file: Filename (for error messages)
    
    Returns:
        tuple: (list_of_points, error) where list_of_points is a list of lists
               and error is a list of error messages (empty if no errors)
    """
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
    """
    Calculate Euclidean distance between two n-dimensional points.
    
    Args:
        n: Dimension of the points
        first_individual: First point (list of floats)
        second_individual: Second point (list of floats)
    
    Returns:
        float: Euclidean distance
    """
    distance = 0
    for i in range(0, n):
        distance = distance + (first_individual[i] - second_individual[i])**2
    distance = math.sqrt(distance)
    return distance


def location_of_GraphFiles(problem_code, graph_digit):
    """
    Get the file path for a graph input file.
    
    Args:
        problem_code: Problem type code (e.g., 'GP', 'GC', 'CL')
        graph_digit: Graph variant (e.g., 'A', 'B', 'C')
    
    Returns:
        str: Path to graph file
    """
    input_file = os.path.join("GraphFiles", problem_code + "Graph" + graph_digit + ".txt")
    return input_file


def location_of_witness_set(graph_digit, timestamp):
    """
    Generate file path for witness set output in outputs/ directory.
    
    Args:
        graph_digit: Graph variant (e.g., 'A', 'B', 'C')
        timestamp: Timestamp string
    
    Returns:
        str: Full path to witness set file in outputs/
    """
    # Get the directory containing utils.py (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get project root (parent of nchw73/)
    project_root = os.path.dirname(script_dir)
    # Build path to outputs/
    witness_set = os.path.join(project_root, "outputs", "Witness" + graph_digit + "_" + timestamp + ".txt")
    return witness_set
