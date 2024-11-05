#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the validation program for your submission.
# This program should sit in a folder along with the folder 'abcd12' (substitute
# with your username), which contains everything that you intend to submit, and
# the folder 'GraphFiles' containing the graph data files as I supplied to you.

# Run this program and when prompted, enter your username.

# A validation file is automatically placed in the folder 'abcd12' and the contents
# of this file are printed to the screen too. Do not remove this validation file 
# from the folder 'abcd12'. So prior to zipping, you should ensure that the
# validation file 'validation.txt' is included in the folder 'abcd12'.

# Immediately prior to submitting, run this program a final time. Check the
# validation file and if all is well then you can submit. Otherwise, fix what
# needs to be fixed and go again.

#####################################
#### DO NOT TOUCH ANYTHING BELOW ####
#####################################

legal_modules = ['abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore', 'atexit', 'audioop',
                 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins', 'bz2',
                 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections', 'colorsys',
                     'compileall', 'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg', 'CProfile',
                     'crypt', 'csv', 'ctypes', 'curses',
                 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'dummy_threading',
                 'email', 'encodings', 'ensurepip', 'enum', 'errno',
                 'faulthandler', 'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib', 'functools',
                 'gc', 'getopt', 'getpass', 'gettext', 'glob', 'grp', 'gzip',
                 'hashlib', 'heapq', 'hmac', 'html', 'http',
                 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
                 'json',
                 'keyword',
                 'lib2to3', 'linecache', 'locale', 'logging', 'lzma',
                 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'msilib', 'msvcrt', 'multiprocessing',
                 'netrc', 'nis', 'nntplib', 'numbers', 'numpy',
                 'operator', 'optparse', 'os', 'ossaudiodev',
                 'parser', 'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix',
                     'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc',
                 'queue', 'quopri',
                 'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy',
                 'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr',
                     'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess',
                     'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog',
                 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter',
                     'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing',
                 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid',
                 'venv',
                 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound', 'wsgiref',
                 'xdrlib', 'xml', 'xmlrpc',
                 'zipapp', 'zipfile', 'zipimport', 'zlib']

import sys
import os
import math

username_NegSel_tag = '##############################\n'
username_NegSel_tag = username_NegSel_tag + '#### ENTER YOUR USER-NAME ####\n'
username_NegSel_tag = username_NegSel_tag + '##############################\n\nusername = "'
len_username_NegSel_tag = len(username_NegSel_tag)
alg_code_NegSel_tag = '###############################################################\n'
alg_code_NegSel_tag = alg_code_NegSel_tag + '#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####\n'
alg_code_NegSel_tag = alg_code_NegSel_tag + '###############################################################\n\nalg_code = "'
len_alg_code_NegSel_tag = len(alg_code_NegSel_tag)
start_of_threshold_tag = "#####################################################################################################################\n"
start_of_threshold_tag = start_of_threshold_tag + "#### ENTER THE THRESHOLD: IF YOU ARE IMPLEMENTING VDETECTOR THEN SET THE THRESHOLD AS YOUR CHOICE OF SELF-RADIUS ####\n"
start_of_threshold_tag = start_of_threshold_tag + "#####################################################################################################################\n"
start_of_threshold_tag = start_of_threshold_tag + "\nthreshold = "
len_start_of_threshold_tag = len(start_of_threshold_tag)
start_of_num_detectors_tag = "######################################################\n"
start_of_num_detectors_tag = start_of_num_detectors_tag + "#### ENTER THE INTENDED SIZE OF YOUR DETECTOR SET ####\n"
start_of_num_detectors_tag = start_of_num_detectors_tag + "######################################################\n"
start_of_num_detectors_tag = start_of_num_detectors_tag + "\nnum_detectors = "
len_start_of_num_detectors_tag = len(start_of_num_detectors_tag)

username_NatAlgReal_tag = '##############################\n'
username_NatAlgReal_tag = username_NatAlgReal_tag + '#### ENTER YOUR USER-NAME ####\n'
username_NatAlgReal_tag = username_NatAlgReal_tag + '##############################\n\nusername = "'
len_username_NatAlgReal_tag = len(username_NatAlgReal_tag)
alg_code_NatAlgReal_tag = '###############################################################\n'
alg_code_NatAlgReal_tag = alg_code_NatAlgReal_tag + '#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####\n'
alg_code_NatAlgReal_tag = alg_code_NatAlgReal_tag + '###############################################################\n\nalg_code = "'
len_alg_code_NatAlgReal_tag = len(alg_code_NatAlgReal_tag)

username_NatAlgDiscrete_tag = '##############################\n'
username_NatAlgDiscrete_tag = username_NatAlgDiscrete_tag + '#### ENTER YOUR USER-NAME ####\n'
username_NatAlgDiscrete_tag = username_NatAlgDiscrete_tag + '##############################\n\nusername = "'
len_username_NatAlgDiscrete_tag = len(username_NatAlgDiscrete_tag)
alg_code_NatAlgDiscrete_tag = '###############################################################\n'
alg_code_NatAlgDiscrete_tag = alg_code_NatAlgDiscrete_tag + '#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####\n'
alg_code_NatAlgDiscrete_tag = alg_code_NatAlgDiscrete_tag + '###############################################################\n\nalg_code = "'
len_alg_code_NatAlgDiscrete_tag = len(alg_code_NatAlgDiscrete_tag)
problem_code_NatAlgDiscrete_tag = '#################################################################\n'
problem_code_NatAlgDiscrete_tag = problem_code_NatAlgDiscrete_tag + '#### ENTER THE CODE FOR THE GRAPH PROBLEM YOU ARE OPTIMIZING ####\n'
problem_code_NatAlgDiscrete_tag = problem_code_NatAlgDiscrete_tag + '#################################################################\n\nproblem_code = "'
len_problem_code_NatAlgDiscrete_tag = len(problem_code_NatAlgDiscrete_tag)
graph_digit_NatAlgDiscrete_tag = '#############################################################\n'
graph_digit_NatAlgDiscrete_tag = graph_digit_NatAlgDiscrete_tag + '#### ENTER THE DIGIT OF THE INPUT GRAPH FILE (A, B OR C) ####\n'
graph_digit_NatAlgDiscrete_tag = graph_digit_NatAlgDiscrete_tag + '#############################################################\n\ngraph_digit = "'
len_graph_digit_NatAlgDiscrete_tag = len(graph_digit_NatAlgDiscrete_tag)

tag = []
tag.append("username = ")
tag.append("problem code = ")
tag.append("graph = ")
tag.append("colours to use = ")
tag.append("number of partition sets = ")
tag.append("algorithm code = ")
tag.append("associated parameters [n, num_cyc, N, M, lambbda] = ")
tag.append("associated parameters [n, num_cyc, N, lambbda, alpha] = ")
tag.append("associated parameters [n, num_cyc, N, p, q, alpha, beta] = ")
tag.append("associated parameters [n, num_cyc, N, b] = ")
tag.append("associated parameters [n, num_cyc, sigma, f_max, f_min] = ")
tag.append("conflicts = ")
tag.append("clique size = ")
tag.append("elapsed time = ")
tag.append("certificate = ")
tag_length = []
for i in range(0,len(tag)):
    tag_length.append(len(tag[i]))

parameter_names =  {'AB':['n', 'num_cyc', 'N', 'M', 'lambbda'], 'FF':['n', 'num_cyc', 'N', 'lambbda', 'alpha'], 'CS':['n', 'num_cyc', 'N', 'p', 'q', 'alpha', 'beta'],
                        'WO':['n', 'num_cyc', 'N', 'b'], 'BA':['n', 'num_cyc', 'sigma', 'f_max', 'f_min']}
parameter_tag = {'AB':6, 'FF':7, 'CS':8, 'WO':9, 'BA':10}
parameter_float = {'AB':4, 'FF':3, 'CS':3, 'WO':3, 'BA':2}

def get_graph_data(graph_file, problem_code):
    # read a graph file and pull out the core data

    # set up the tags and tag-lengths
    
    vertices_tag = "number of vertices = "
    len_vertices_tag = len(vertices_tag)
    edges_tag = "number of edges = "
    len_edges_tag = len(edges_tag)
    colours_to_use_tag = "number of colours to use = "
    len_colours_to_use_tag = len(colours_to_use_tag)
    sets_in_partition_tag = "number of partition sets = "
    len_sets_in_partition_tag = len(sets_in_partition_tag)

    # build full location of input file and open it
        
    f = open(graph_file, "r")

    # get the number of vertices
    
    whole_line = f.readline()
    vertices = whole_line[len_vertices_tag:len(whole_line) - 1]
    v = int(vertices)

    # get the number of edges
    
    whole_line = f.readline()
    edges = whole_line[len_edges_tag:len(whole_line) - 1]
    edges = int(edges)

    # get the number of colours/partition sets if appropriate
    
    if problem_code == "GC":
        whole_line = f.readline()
        colours_to_use = whole_line[len_colours_to_use_tag:len(whole_line) - 1]
        colours_to_use = int(colours_to_use)
    if problem_code == "GP":
        whole_line = f.readline()
        sets_in_partition = whole_line[len_sets_in_partition_tag:len(whole_line) - 1]
        sets_in_partition = int(sets_in_partition)

    # read the adjacency matrix (it is upper triangular form with no leading diagonals)
    # every list of vertex adjacencies sits on one line, terminated with a comma and an EOL
    # except for the final line which can be terminated with either a comma and an EOL or
    # just an EOL
    
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
            if splitline[len(splitline) - 1] == ",":
                splitline = splitline[0:len(splitline) - 1]
            splitline = [int(splitline)]
        splitline.insert(0, 0)
        matrix.append(splitline[:])            
    matrix.append([0])
    for i in range(0, v):
        for j in range(0, i):
            matrix[i].insert(j, matrix[j][i])
            
    f.close()

    # return data depending upon the problem code (we are not bothered about the number of edges)

    if problem_code == "GC":
        return v, matrix, colours_to_use
    elif problem_code == "GP":
        return v, matrix, sets_in_partition
    else:
        return v, matrix

def check_parameters(whole_line, list_of_errors, raw_witness_file, read_alg_code, parameter_names, parameter_tag, parameter_float):
    if read_alg_code in ['AB', 'FF', 'CS', 'WO', 'BA']:
        parameters = whole_line[tag_length[parameter_tag[read_alg_code]] + 1:len(whole_line) - 2]
        parameters = parameters.split(",")
        length = len(parameters)
        if length == len(parameter_names[read_alg_code]):
            for i in range(0, length):
                if i < parameter_float[read_alg_code]:
                    try:
                        parameters[i] = int(parameters[i])
                    except:
                        list_of_errors.append("*** error: the algorithm code from '{0}' is '{1}' yet the value of parameter '{2}', namely '{3}', is not an integer\n".format(raw_witness_file, read_alg_code, parameter_names[read_alg_code][i], parameters[i]))
                        parameters[i] = -1
                else:
                    try:
                        parameters[i] = float(parameters[i])
                    except:
                        list_of_errors.append("*** error: the algorithm code from '{0}' is '{1}' yet the value of parameter '{2}', namely '{3}', is not a float\n".format(raw_witness_file, read_alg_code, parameter_names[read_alg_code][i], parameters[i]))
                        parameters[i] = -1   
        else:
            list_of_errors.append("*** error: the algorithm code '{0}' is '{1}' yet there are an incorrect number {1} of parameters\n".format(raw_witness_file, read_alg_code, length))
            parameters = []
            for i in range(0, len(parameter_names[read_alg_code])):
                parameters.append(-1)
    else:
        parameters = []
        for i in range(0, len(parameter_names[read_alg_code])):
            parameters.append(-1)
    return parameters, list_of_errors

def check_witness_file(username, graph_file, witness_file, tag, tag_length, problem_code_from_NatAlgDiscrete, alg_code_NatAlgDiscrete, graph_digit_NatAlgDiscrete, witness_digit, parameter_names, parameter_tag, parameter_float):
    # validate a witness file - both 'graph_file' and 'witness_file' are full paths to the files
    # both files are assumed to exist and errors are collected (all errors are fatal errors)

    list_of_errors = []
    witness_data = []
    
    # read graph data depending upon the problem code
    
    if problem_code_from_NatAlgDiscrete == "GC":
        v, matrix, colours_to_use = get_graph_data(graph_file, problem_code_from_NatAlgDiscrete)
    elif problem_code_from_NatAlgDiscrete == "GP":
        v, matrix, sets_in_partition = get_graph_data(graph_file, problem_code_from_NatAlgDiscrete)
    elif problem_code_from_NatAlgDiscrete == "CL":
        v, matrix = get_graph_data(graph_file, problem_code_from_NatAlgDiscrete)

    # open the witness file
    
    f = open(witness_file, "r")
    raw_witness_file = witness_file[len(witness_file) - 12:len(witness_file)]

    # check that the username is valid (tag[0])

    whole_line = f.readline()
    read_username = whole_line[tag_length[0]:len(whole_line) - 1]
    if read_username != username:
        list_of_errors.append("*** error: the username '{0}' read from '{1}' is different to the stated username '{2}'\n".format(read_username, raw_witness_file, username))

    # get the claimed problem code (tag[1])

    whole_line = f.readline()
    read_problem_code = whole_line[tag_length[1]:tag_length[1] + 2]
    if not read_problem_code in ["GC", "CL", "GP"]:
        list_of_errors.append("*** error: the problem code '{0}' in '{1}' is invalid\n".format(read_problem_code, raw_witness_file))
    elif read_problem_code != problem_code_from_NatAlgDiscrete:
        list_of_errors.append("*** error: the problem code '{0}' in '{1}' is different to the problem code '{2}' from 'NatAlgDiscrete{3}.py'\n".format(read_problem_code, raw_witness_file, problem_code_from_NatAlgDiscrete, witness_digit))
    
    # get the claimed graph file (tag[2])
    
    whole_line = f.readline()
    read_graph_file = whole_line[tag_length[2]:tag_length[2] + 12]
    stated_graph_digit = read_graph_file[7]
    given_graph_file = problem_code_from_NatAlgDiscrete + "Graph" + witness_digit + ".txt"
    if read_graph_file != given_graph_file:
        list_of_errors.append("*** error: there is a mismatch between the graph file '{0}' from '{1}' and the given graph file '{2}'\n".format(read_graph_file, raw_witness_file, given_graph_file))

    # if problem code is graph colouring then check number of colours used (tag[3])
    
    if read_problem_code == "GC":
        whole_line = f.readline()
        read_colours_to_use = whole_line[tag_length[3]:len(whole_line) - 1]
        try:
            read_colours_to_use = int(read_colours_to_use)
            if read_colours_to_use > colours_to_use:
                list_of_errors.append("*** error: the colours used [1...{0}] from '{1}' do not match with those allowed [1...{2}]\n".format(read_colours_to_use, raw_witness_file, colours_to_use))
        except:
            list_of_errors.append("*** error: the colours used [1...{0}] from '{1}' is invalid\n".format(read_colours_to_use, raw_witness_file))
            read_colours_to_use = 0

    # if problem code is graph partitioning then check number of partition sets (tag[4])

    if read_problem_code == "GP":
        whole_line = f.readline()
        read_partition_sets = whole_line[tag_length[4]:len(whole_line) - 1]
        try:
            read_partition_sets = int(read_partition_sets)
            if read_partition_sets != sets_in_partition:
                list_of_errors.append("*** error: the number of sets partitioned into, namely {0}, from '{1}' does not match with that reqested, namely {2}\n".format(read_partition_sets, raw_witness_file, sets_in_partition))
        except:
            list_of_errors.append("*** error: the number of sets partitioned into, namely {0}, from '{1}' is invalid\n".format(read_partition_sets, raw_witness_file))
            read_partition_sets = 0

    # check that the algorithm code is valid (tag[5])
    
    whole_line = f.readline()
    read_alg_code = whole_line[tag_length[5]:len(whole_line) - 1]
    if not read_alg_code in ["AB", "FF", "CS", "WO", "BA"]:
        list_of_errors.append("*** error: the algorithm code '{0}' from '{1}' is invalid\n".format(read_alg_code, raw_witness_file))
    elif read_alg_code != alg_code_NatAlgDiscrete:
        list_of_errors.append("*** error: the algorithm code '{0}' from '{1}' is different to that from 'NatAlgDisc{2}.py' namely '{3}'\n".format(read_alg_code, raw_witness_file, witness_digit, alg_code_NatAlgDiscrete))

    # check that the associated parameters match the algorithm code (tag[6-10])
    
    whole_line = f.readline()
    if (read_alg_code == "AB" and whole_line[0:tag_length[6]] != tag[6]) or (read_alg_code == "FF" and whole_line[0:tag_length[7]] != tag[7]) or (
        read_alg_code == "CS" and whole_line[0:tag_length[8]] != tag[8]) or (read_alg_code == "WO" and whole_line[0:tag_length[9]] != tag[9]) or (
        read_alg_code == "BA" and whole_line[0:tag_length[10]] != tag[10]):
        list_of_errors.append("*** error: the associated parameters from '{0}' are incorrect for the stated algorithm code '{1}'\n".format(raw_witness_file, read_alg_code))
        error_flag = True
    parameters, list_of_errors = check_parameters(whole_line, list_of_errors, raw_witness_file, read_alg_code, parameter_names, parameter_tag, parameter_float)
    
    # get the claimed conflicts/clique size (tag[11-12])
    
    whole_line = f.readline()
    if read_problem_code == "GC" or read_problem_code == "GP":
        read_conflicts = whole_line[tag_length[11]:len(whole_line) - 1]
        try:
            read_conflicts = int(float(read_conflicts))
        except:
            list_of_errors.append("*** error: the number of conflicts from '{0}', namely '{1}', is not integral\n".format(raw_witness_file, read_conflicts))
            read_conflicts = -1
    elif read_problem_code == "CL":
        read_clique_size = whole_line[tag_length[12]:len(whole_line) - 1]
        try:
            read_clique_size = int(float(read_clique_size))
        except:
            list_of_errors.append("*** error: the clique size from '{0}', namely '{1}', is not integral\n".format(raw_witness_file, read_clique_size))
            read_conflicts = -1

    # get the elapsed time (tag[13])
    
    whole_line = f.readline()
    elapsed_time = whole_line[tag_length[13]:len(whole_line) - 1]
    try:
        elapsed_time = round(float(elapsed_time), 1)
    except:
        list_of_errors.append("*** error: the elapsed time from '{0}', namely '{1}', is not float\n".format(raw_witness_file, elapsed_time))
        elapsed_time = -1

    # get certificate (tag[14])

    whole_line = f.readline()
    whole_line = f.readline()
    read_certificate = whole_line[tag_length[14]:len(whole_line) - 1]
    try:
        read_certificate = int(read_certificate)
    except:
        list_of_errors.append("*** error: the certificate '{0}', namely '{1}', is not integral\n".format(raw_witness_file, read_certificate))
        elapsed_time = -1

    # set up witnessing data structure
    
    if read_problem_code == "GC":
        colouring = []
    elif read_problem_code == "CL":
        clique = []
    elif read_problem_code == "GP":
        partition = []

    # build witnessing data structure (note that a clique, for example, might be written as a number of lines
    # of comma-separated entries: it does not matter if there is a comma after the last entry or not)
    
    whole_line = f.readline()
    bad_digit = False
    end_error_tag = "*** ERRORS: There were errors"
    len_end_error_tag = len(end_error_tag)
    while whole_line != "" and whole_line[0:len_end_error_tag] != end_error_tag:
        whole_line = whole_line[0:len(whole_line) - 1]
        if whole_line[len(whole_line) - 1] == ",":
            whole_line = whole_line[0:len(whole_line) - 1]
        line_list = whole_line.split(",")
        line_list = line_list[0:len(line_list)]
        length = len(line_list)
        for i in range(0, length):
            try:
                the_digit = int(float(line_list[i]))
            except:
                bad_digit = True
                the_digit = -1
            if read_problem_code == "GC":
                colouring.append(the_digit)
            elif read_problem_code == "CL":
                clique.append(the_digit)
            elif read_problem_code == "GP":
                partition.append(the_digit)
        whole_line = f.readline()
        
    if bad_digit == True:
        if read_problem_code == "GC":
            list_of_errors.append("*** error: there is an invalid entry in the colouring from '{0}'\n".format(raw_witness_file))
        elif read_problem_code == "CL":
            list_of_errors.append("*** error: there is an invalid entry in the clique from '{0}'\n".format(raw_witness_file))
        elif read_problem_code == "GP":
            list_of_errors.append("*** error: there is an invalid entry in the partition from '{0}'\n".format(raw_witness_file))

    if whole_line[0:len_end_error_tag] == end_error_tag:
        whole_line = f.readline()
        list_of_errors.append("*** there were additional errors already in '{0}':\n".format(raw_witness_file))
        while whole_line != "":
            list_of_errors.append(whole_line)
            whole_line = f.readline()
        
    if read_problem_code == "GC":

        # check that the colouring colours every vertex
        
        length = len(colouring)
        if length != v:
            list_of_errors.append("*** error: the colouring from '{0}' has length {1} but should have length {2}\n".format(raw_witness_file, length, v))
            read_conflicts = -1
        else:

        # check that no illegal colours are used
        
            bad_colouring = False
            for i in range(0, length):
                if colouring[i] < 1 or colouring[i] > colours_to_use:
                    bad_colouring = True
                    break
            if bad_colouring == True:
                list_of_errors.append("*** error: the colouring from '{0}' contains illegal colours\n".format(raw_witness_file))
                read_conflicts = -1
            else:

                # check that the number of stated conflicts (bad edges) is correct
        
                true_conflicts = 0
                for i in range(0, length):
                    for j in range(i + 1, length):
                        if matrix[i][j] == 1 and colouring[i] == colouring[j]:
                            true_conflicts = true_conflicts + 1
                try:
                    read_conflicts = int(float(read_conflicts))
                    if read_conflicts != true_conflicts:
                        list_of_errors.append("*** error: the number of colour conflicts {0} from '{1}' differs from the actual number of conflicts {2}\n".format(read_conflicts, raw_witness_file, true_conflicts))
                        read_conflicts = -1
                except:
                    list_of_errors.append("*** error: the number of colour conflicts {0} from '{1}' is invalid\n".format(read_conflicts, raw_witness_file))
                    read_conflicts = -1
        
    elif read_problem_code == "CL":

        # check that the clique is a list of the correct number of vertices
        
        length = len(clique)
        if length != v:
            list_of_errors.append("*** error: the clique string from '{0}' has length {1} but should have length {2}\n".format(raw_witness_file, length, v))
            read_clique_size = -1
        else:
            
            # check that the list contains only 0s and 1s
            
            bad_clique = False
            for i in range(0, length):
                if clique[i] != 0 and clique[i] != 1:
                    bad_clique = True
                    bad_character = clique[i]
                    break
            if bad_clique == True:
                list_of_errors.append("*** error: the clique string from '{0}' is not a list of 0s and 1s\n".format(raw_witness_file))
                read_clique_size = -1
            else:
                
                # check that the clique has the size stated
                
                true_size = 0
                for i in range(0, length):
                    if clique[i] == 1:
                        true_size = true_size + 1
                try:
                    read_clique_size = int(read_clique_size)
                    if read_clique_size != true_size:
                        list_of_errors.append("*** error: the clique from '{0}' is claimed to have size {1} but the list contains {2} vertices\n".format(raw_witness_file, read_clique_size, true_size))
                        read_clique_size = -1
                    else:
                        bad_edges = 0
                        bad_clique = False
                        for i in range(0, length):
                            for j in range(i + 1, length):
                                if clique[i] == 1 and clique[j] == 1 and matrix[i][j] != 1:
                                    bad_edges = bad_edges + 1
                                    bad_clique = True
                        if bad_clique == True:
                            list_of_errors.append("*** error: the clique from '{0}' of claimed size {1} is not a clique as there are {2} missing edges\n".format(raw_witness_file, read_clique_size, bad_edges))
                            read_clique_size = -1                    
                except:
                    list_of_errors.append("*** error: the clique size {0} from '{1}' is invalid\n".format(read_clique_size, raw_witness_file))
                    read_clique_size = -1

    elif read_problem_code == "GP":

        # check that every vertex appears in a partite set
        
        length = len(partition)
        if length != v:
            list_of_errors.append("*** error: the partition string from '{0}' has length {1} but should have length {2}\n".format(raw_witness_file, length, v))
            read_conflicts = -1
        else:
            
            # check that the correct set names are used (1..'sets_in_partition')
            
            bad_partition = False
            for i in range(0, length):
                if partition[i] < 1 and partition[i] > sets_in_partition:
                    bad_partition = True
                    bad_partition_character = partition[i]
                    break
            if bad_partition == True:
                list_of_errors.append("*** error: the partition string from '{0}' is not a list of integers from [1..{0}]\n".format(raw_witness_file, sets_in_partition))
                read_conflicts = -1
            else:
                
                # check that the partition is legal
                
                sizes_of_sets = []
                for i in range(0, sets_in_partition):
                    sizes_of_sets.append(0)    
                for i in range(0, length):
                    sizes_of_sets[partition[i] - 1] = sizes_of_sets[partition[i] - 1] + 1
                for i in range(0, sets_in_partition):
                    if sizes_of_sets[i] != math.floor(v / sets_in_partition) and sizes_of_sets[i] != math.floor(v / sets_in_partition) + 1:
                        bad_partition = True
                        break
                if bad_partition == True:
                    list_of_errors.append("*** error: the partition string from '{0}' does not describe a legal partition\n".format(raw_witness_file))
                    read_conflicts = -1
                else:
                    
                    # check the number of conflicts, i.e., cross-partite-sets edges
                    
                    true_conflicts = 0
                    for i in range(0, length):
                        for j in range(i + 1, length):
                            if matrix[i][j] == 1 and partition[i] != partition[j]:
                                true_conflicts = true_conflicts + 1
                    try:
                        read_conflicts = int(read_conflicts)
                        if read_conflicts != true_conflicts:
                            list_of_errors.append("*** error: the number of partition conflicts {0} from '{1}' differs from the actual number of conflicts {2}\n".format(read_conflicts, raw_witness_file, true_conflicts))
                            read_conflicts = -1
                    except:
                        list_of_errors.append("*** error: the number of partition conflicts {0} from '{1}' is invalid\n".format(read_conflicts, raw_witness_file))
                        read_conflicts = -1
        
    f.close()

    # store the 13 entries in the record

    if read_problem_code == "GC" or read_problem_code == "GP":
        witness_data = [username, read_alg_code, read_problem_code, stated_graph_digit, read_conflicts, elapsed_time]
    elif read_problem_code == "CL":
        witness_data = [username, read_alg_code, read_problem_code, stated_graph_digit, read_clique_size, elapsed_time]

    # store the parameters, padded with 'X's
    
    num_parameters = len(parameters)

    for i in range(0, 7):
        if i < num_parameters:
            witness_data.append(parameters[i])
        else:
            witness_data.append("X")

    # note that if there is no error then 'list_of_errors' is returned as '[]'
    
    return list_of_errors, witness_data

def remove_all_spaces(the_string):
    # remove all spaces from a string
    length = len(the_string)
    new_string = ""
    for i in range(length):
        if the_string[i] != " ":
            new_string = new_string + the_string[i]
    return new_string

def read_file_into_string(file_location):
    f = open(file_location, mode = 'r', encoding = 'utf-8')
    the_string = ""
    x = f.read(1)
    while x != "":
        the_string = the_string + x
        x = f.read(1)
    f.close()
    return the_string

def get_the_modules_imported(program_file_string):
    # get the modules imported in a program that has been converted to a string
    import_sandwiches = []
    location = 0
    if program_file_string[0:6] == "import":
        location = program_file_string.find("\n")
        if location != -1:
            import_sandwiches.append(program_file_string[6:location])
        else:
            import_sandwiches.append(program_file_string[6:])
            location = len(program_file_string)
    found = 0
    while found != -1:
        found = program_file_string.find("\nimport", location)
        if found != -1:
            location = program_file_string.find("\n", found + 7)
            if location != -1:
                import_sandwiches.append(program_file_string[found + 7:location])
            else:
                import_sandwiches.append(program_file_string[found + 7:])
                found = -1
    from_sandwiches = []
    location = 0
    if program_file_string[0:4] == "from":
        location = program_file_string.find("\n")
        if location != -1:
            import_sandwiches.append(program_file_string[4:location])
        else:
            import_sandwiches.append(program_file_string[4:])
            location = len(program_file_string)
        from_sandwiches.append(program_file_string[4:location])
    found = 0
    while found != -1:
        found = program_file_string.find("\nfrom", location)
        if found != -1:
            location = program_file_string.find("\n", found + 5)
            if location != -1:
                from_sandwiches.append(program_file_string[found + 5:location])
            else:
                from_sandwiches.append(program_file_string[found + 5:])
                found = -1       
    the_imports = []
    for item in import_sandwiches:
        found, location = 0, 0
        while found != -1:
            found = item.find(",", location)
            if found != -1:
                as_found = item.find(" as ", location, found)
                if as_found != -1:
                    the_imports.append(remove_all_spaces(item[location:as_found]))
                else:
                    the_imports.append(remove_all_spaces(item[location:found]))
                location = found + 1
            else:
                as_found = item.find(" as ", location)
                if as_found != -1:
                    the_imports.append(remove_all_spaces(item[location:as_found]))
                else:
                    the_imports.append(remove_all_spaces(item[location:]))
    for item in from_sandwiches:
        found = item.find("import")
        if found != -1:
            the_imports.append(remove_all_spaces(item[0:found]))      
    length = len(the_imports)
    for i in range(0, length):
        found = the_imports[i].find(".")
        if found != -1:
            the_imports[i] = the_imports[i][0:found]

    the_imports = list(set(the_imports))
    the_imports.sort()    
    return the_imports

def validate_witness_file(path_to_witness, path_to_graph, tag, tag_length, problem_code_NatAlgDiscrete, alg_code_NatAlgDiscrete, graph_digit_NatAlgDiscrete, witness_digit, g, parameter_names, parameter_tag, parameter_float):
    # Validate the witness file at 'path_to_witness', which is assumed to exist, using 'path_to_graph', which is also assumed to exist.
    # 'witness_digit' is the digit of the witness file built by 'NatAlgDiscrete<witness>.py' 
    # problem_code_NatAlgDiscrete = problem code from 'NatAlgDiscrete<witness_digit>.py'
    # alg_code_NatAlgDiscrete = algorithm code from 'NatAlgDiscrete<witness_digit>.py'
    # graph_digit_NatAlgDiscrete = graph digit from 'NatAlgDiscrete<witness_digit>.py'

    g.write("for 'Witness{0}.txt':\n".format(witness_digit))
    list_of_errors, witness_data = check_witness_file(username, path_to_graph, path_to_witness, tag, tag_length, problem_code_NatAlgDiscrete, alg_code_NatAlgDiscrete, graph_digit_NatAlgDiscrete, witness_digit, parameter_names, parameter_tag, parameter_float)
    number_of_errors = len(list_of_errors)
    g.write("  username = '{0}'; algorithm code = '{1}'; problem code = '{2}'; graph digit = '{3}'\n".format(witness_data[0], witness_data[1], witness_data[2], witness_data[3]))
    if witness_data[1] == "AB":
        g.write("  parameters = [n = {0}, num_cyc = {1}, N = {2}, M = {3}, lambbda = {4}]\n".format(witness_data[6], witness_data[7], witness_data[8], witness_data[9], witness_data[10]))
    elif witness_data[1] == "FF":
        g.write("  parameters = [n = {0}, num_cyc = {1}, N = {2}, lambbda = {3}, alpha = {4}]\n".format(witness_data[6], witness_data[7], witness_data[8], witness_data[9], witness_data[10]))
    elif witness_data[1] == "CS":
        g.write("  parameters = [n = {0}, num_cyc = {1}, N = {2}, p = {3}, q = {4}, alpha = {5}, beta = {6}]\n".format(witness_data[6], witness_data[7], witness_data[8], witness_data[9], witness_data[10], witness_data[11], witness_data[12]))
    elif witness_data[1] == "WO":
        g.write("  parameters = [n = {0}, num_cyc = {1}, N = {2}, b = {3}]\n".format(witness_data[6], witness_data[7], witness_data[8], witness_data[9]))
    elif witness_data[1] == "BA":
        g.write("  parameters = [n = {0}, num_cyc = {1}, sigma = {2}, f_max = {3}, f_min = {4}]\n".format(witness_data[6], witness_data[7], witness_data[8], witness_data[9], witness_data[10]))
    if witness_data[2] == "GC" or witness_data[2] == "GP":
        g.write("  conflicts = {0}; elapsed time = {1}\n".format(witness_data[4], witness_data[5]))
    elif witness_data[2] == "CL":
        g.write("  clique size = {0}; elapsed time = {1}\n".format(witness_data[4], witness_data[5]))
    for i in range(0, number_of_errors):
        g.write("{0}".format(list_of_errors[i]))
    return list_of_errors

def read_NatAlgDiscrete_file(path_to_student_folder, program_digit, username_NatAlgDiscrete_tag, len_username_NatAlgDiscrete_tag, alg_code_NatAlgDiscrete_tag, len_alg_code_NatAlgDiscrete_tag,
                             problem_code_NatAlgDiscrete_tag, len_problem_code_NatAlgDiscrete_tag, graph_digit_NatAlgDiscrete_tag, len_graph_digit_NatAlgDiscrete_tag, g, error_flag):
    
    current_path = os.path.join(path_to_student_folder, "NatAlgDiscrete" + program_digit + ".py")
    NatAlgDiscrete_string = read_file_into_string(current_path)

    # get user-name, algorithm code, problem code and graph digit from NatAlgDiscrete?.py
    
    start = NatAlgDiscrete_string.find(username_NatAlgDiscrete_tag) + len_username_NatAlgDiscrete_tag
    end = NatAlgDiscrete_string.find('"', start)
    username_NatAlgDiscrete = NatAlgDiscrete_string[start:end]
    start = NatAlgDiscrete_string.find(alg_code_NatAlgDiscrete_tag) + len_alg_code_NatAlgDiscrete_tag
    end = start + 2
    alg_code_NatAlgDiscrete = NatAlgDiscrete_string[start:end]
    start = NatAlgDiscrete_string.find(problem_code_NatAlgDiscrete_tag) + len_problem_code_NatAlgDiscrete_tag
    end = start + 2
    problem_code_NatAlgDiscrete = NatAlgDiscrete_string[start:end]
    start = NatAlgDiscrete_string.find(graph_digit_NatAlgDiscrete_tag) + len_graph_digit_NatAlgDiscrete_tag
    end = start + 1
    graph_digit_NatAlgDiscrete = NatAlgDiscrete_string[start:end]

    # signal a bad username, a bad algorithm code, a bad problem code or a bad graph digit

    g.write("for 'NatAlgDiscrete{0}.py':\n".format(program_digit))
    g.write("  username = '{0}'; algorithm code = '{1}'; problem code = '{2}'; graph digit = '{3}'\n".format(username_NatAlgDiscrete, alg_code_NatAlgDiscrete, problem_code_NatAlgDiscrete, graph_digit_NatAlgDiscrete))
    if username_NatAlgDiscrete != username:
        g.write("*** error: the username '{0}' in 'NatAlgDiscrete{1}.py' is different from the correct username '{2}'\n".format(username_NatAlgDiscrete, program_digit, username))
        error_flag = True
    if not alg_code_NatAlgDiscrete in ["AB", "FF", "CS", "WO", "BA"]:
        g.write("*** error: the algorithm code '{0}' in 'NatAlgDiscrete{1}.py' is invalid\n".format(alg_code_NatAlgDiscrete, program_digit))
        error_flag = True
    if not problem_code_NatAlgDiscrete in ["GC", "CL", "GP"]:
        g.write("*** error: the problem code '{0}' in 'NatAlgDiscrete{1}.py' is invalid\n".format(problem_code_NatAlgDiscrete, program_digit))
        error_flag = True
    if graph_digit_NatAlgDiscrete != program_digit or not problem_code_NatAlgDiscrete in ["GC", "CL", "GP"]:
        g.write("*** error: graph '{0}Graph{1}' in 'NatAlgDiscrete{2}.py' is the wrong input graph\n".format(problem_code_NatAlgDiscrete, graph_digit_NatAlgDiscrete, program_digit))
        error_flag = True

    # signal bad modules imported
        
    NatAlgDiscrete_imports = get_the_modules_imported(NatAlgDiscrete_string)
    for item in NatAlgDiscrete_imports:
        if not item in legal_modules:
            g.write("*** error: 'NatAlgDiscrete{1}.py' imports the illegal module '{0}'\n".format(item, program_digit))
            error_flag = True
    return alg_code_NatAlgDiscrete, problem_code_NatAlgDiscrete, graph_digit_NatAlgDiscrete, error_flag

######################
#### main program ####
######################

if __name__ == "__main__":

    if not os.path.isdir("GraphFiles"):
        print("*** error: there does not exist a folder named 'GraphFiles' that is supposed to contain the graph files")
        sys.exit()
    
    username = input("\nEnter your username followed by <return>: ")
    print("")

    if not os.path.isdir(username):
        print("*** error: there does not exist a folder named '{0}' that is supposed to contain your submission".format(username))
        sys.exit()

    path_to_student_folder = username
    output_file_path = os.path.join(username, "validation.txt")
    g = open(output_file_path, "w")
    g.write("\nValidation of student '{0}':\n".format(username))
    g.write("------------------------------\n\n")
    
    error_flag = False
    warning_flag = False
    
    student_submission = os.listdir(username)

    # these are the items required

    list_of_items = ["NegSelTraining.py", "NatAlgReal.py", "NatAlgDiscreteA.py", "NatAlgDiscreteB.py", "NatAlgDiscreteC.py", \
                         "WitnessA.txt", "WitnessB.txt", "WitnessC.txt", "NegSelReport.pdf", "NatAlgReport.pdf", "validation.txt"]

    # confirm files that should be in the folder

    for item in list_of_items:
        if item in student_submission:
            g.write("'{0}'".format(item))
            for i in range(0, 20 - len(item)):
                g.write(" ")
            g.write("   exists\n")
        else:
            g.write("*** error: '{0}' should be in the folder '{1}' but isn't\n".format(item, username))
            error_flag = True

    # list rogue items
    
    for item in student_submission:
        if not item in list_of_items:
            g.write("*** warning: '{0}' is an illegal item\n".format(item))
            warning_flag = True

    # read NegSelTraining.py (if it is there)

    if "NegSelTraining.py" in student_submission:

        # build a string of contents of NegSelTraining.py

        current_path = os.path.join(username, "NegSelTraining.py")
        NegSelTraining_string = read_file_into_string(current_path)

        # get user-name and algorithm code from NegSelTraining.py

        start = NegSelTraining_string.find(username_NegSel_tag) + len_username_NegSel_tag
        end = NegSelTraining_string.find('"', start)
        username_NegSel = NegSelTraining_string[start:end]
        start = NegSelTraining_string.find(alg_code_NegSel_tag) + len_alg_code_NegSel_tag
        end = start + 2
        alg_code_NegSel = NegSelTraining_string[start:end]
        start_of_threshold = NegSelTraining_string.find(start_of_threshold_tag) + len(start_of_threshold_tag)
        end_of_threshold = NegSelTraining_string.find("\n", start_of_threshold)
        threshold = NegSelTraining_string[start_of_threshold:end_of_threshold]
        start_of_num_detectors = NegSelTraining_string.find(start_of_num_detectors_tag) + len(start_of_num_detectors_tag)
        end_of_num_detectors = NegSelTraining_string.find("\n", start_of_num_detectors)
        intended_num_detectors = NegSelTraining_string[start_of_num_detectors:end_of_num_detectors]

        g.write("\nfor 'NegSelTraining.py':\n")
        g.write("  username = '{0}'; algorithm code = '{1}'\n".format(username_NegSel, alg_code_NegSel))

        # signal a bad username
        
        if username_NegSel != username:
            g.write("*** error: the username '{0}' in 'NegSelTraining.py' is different from the correct username '{1}'\n".format(username_NegSel, username))
            error_flag = True

        # signal a bad algorithm code

        if not alg_code_NegSel in ["NS", "RV", "VD"]:
            g.write("*** error: the algorithm code '{0}' in 'NegSelTraining.py' is invalid\n".format(alg_code_NegSel))
            error_flag = True

        # signal bad threshold value
        
        try:
            threshold = float(threshold)
        except:
            g.write("*** error: the threshold '{0}' in 'NegSelTraining.py' is not numeric\n".format(threshold))
            error_flag = True

        # signal a non-integral number of intended detectors

        try:
            intended_num_detectors = int(intended_num_detectors)
        except:
            g.write("*** error: the number of detectors '{0}' in 'NegSelTraining.py' is not integral\n".format(intended_num_detectors))
            error_flag = True
 
        # signal bad modules imported

        NegSelTraining_imports = get_the_modules_imported(NegSelTraining_string)
        for item in NegSelTraining_imports:
            if not item in legal_modules:
                g.write("*** error: 'NegSelTraining.py' imports the illegal module '{0}'\n".format(item))
                error_flag = True
                
    else:
        g.write("\nfor 'NegSelTraining.py':\n")
        g.write("*** warning: the file 'NegSelTraining.py' does not exist\n")
        warning_flag = True

    # read NegAlgReal.py (if it is there)
    
    if "NatAlgReal.py" in student_submission:

        current_path = os.path.join(username, "NatAlgReal.py")
        NatAlgReal_string = read_file_into_string(current_path)

        # get user-name and algorithm code from NatAlgReal.py

        start = NatAlgReal_string.find(username_NatAlgReal_tag) + len_username_NatAlgReal_tag
        end = NatAlgReal_string.find('"', start)
        username_NatAlgReal = NatAlgReal_string[start:end]
        start = NatAlgReal_string.find(alg_code_NatAlgReal_tag) + len_alg_code_NatAlgReal_tag
        end = start + 2
        alg_code_NatAlgReal = NatAlgReal_string[start:end]

        g.write("for 'NatAlgReal.py':\n")
        g.write("  username = '{0}'; algorithm code = '{1}'\n".format(username_NatAlgReal, alg_code_NatAlgReal))
        
        # signal a bad username
        
        if username_NatAlgReal != username:
            g.write("*** error: the username '{0}' in 'NatAlgReal.py' is different from the correct username '{1}'\n".format(username_NatAlgReal, username))
            error_flag = True

        # signal a bad algorithm code

        if not alg_code_NatAlgReal in ["AB", "FF", "CS", "WO", "BA"]:
            g.write("*** error: the algorithm code '{0}' in 'NatAlgReal.py' is invalid\n".format(alg_code_NatAlgReal))
            error_flag = True

        # signal bad modules imported
    
        NatAlgReal_imports = get_the_modules_imported(NatAlgReal_string)
        for item in NatAlgReal_imports:
            if not item in legal_modules:
                g.write("*** error: 'NatAlgReal.py' imports the illegal module '{0}'\n".format(item))
                error_flag = True
    else:
        g.write("for 'NatAlgReal.py':\n")
        g.write("*** warning: the file 'NatAlgReal.py' does not exist\n")
        warning_flag = True

    # now deal with the NatAlgDiscrete programs
    
    alg_code_NatAlgDiscrete_dict = {}
    problem_code_NatAlgDiscrete_dict = {}
    graph_digit_NatAlgDiscrete_dict = {}
    for program_digit in ['A', 'B', 'C']:
        if "NatAlgDiscrete" + program_digit + ".py" in student_submission:
            alg_code_NatAlgDiscrete, problem_code_NatAlgDiscrete, graph_digit_NatAlgDiscrete, error_flag = read_NatAlgDiscrete_file(path_to_student_folder, program_digit, username_NatAlgDiscrete_tag, len_username_NatAlgDiscrete_tag, 
                                                                                                        alg_code_NatAlgDiscrete_tag, len_alg_code_NatAlgDiscrete_tag, problem_code_NatAlgDiscrete_tag, 
                                                                                                        len_problem_code_NatAlgDiscrete_tag, graph_digit_NatAlgDiscrete_tag, len_graph_digit_NatAlgDiscrete_tag, g, error_flag)
            alg_code_NatAlgDiscrete_dict[program_digit] = alg_code_NatAlgDiscrete
            problem_code_NatAlgDiscrete_dict[program_digit] = problem_code_NatAlgDiscrete
            graph_digit_NatAlgDiscrete_dict[program_digit] = graph_digit_NatAlgDiscrete
        else:
            g.write("for 'NatAlgDiscrete{0}.py':\n".format(program_digit))
            g.write("*** warning: the file 'NatAlgDiscrete{0}.py' does not exist\n".format(program_digit))
            warning_flag = True

    # signal algorithm code or problem code mismatches

    if "NatAlgDiscreteA.py" in student_submission and "NatAlgDiscreteB.py" in student_submission and "NatAlgDiscreteC.py" in student_submission:
        if alg_code_NatAlgDiscrete_dict['A'] != alg_code_NatAlgDiscrete_dict['B'] or alg_code_NatAlgDiscrete_dict['A'] != alg_code_NatAlgDiscrete_dict['C'] or alg_code_NatAlgDiscrete_dict['B'] != alg_code_NatAlgDiscrete_dict['C']:
            g.write("\n*** error: different algorithm codes in 'NatAlgDiscreteA.py', 'NatAlgDiscreteB.py' and 'NatAlgDiscreteC.py'\n")
            error_flag = True
        elif problem_code_NatAlgDiscrete_dict['A'] != problem_code_NatAlgDiscrete_dict['B'] or problem_code_NatAlgDiscrete_dict['A'] != problem_code_NatAlgDiscrete_dict['C'] or problem_code_NatAlgDiscrete_dict['B'] != problem_code_NatAlgDiscrete_dict['C']:
            g.write("\n*** error: different problem codes in 'NatAlgDiscreteA.py', 'NatAlgDiscreteB.py' and 'NatAlgDiscreteC.py'\n")
            error_flag = True
    elif "NatAlgDiscreteA.py" in student_submission and "NatAlgDiscreteB.py" in student_submission and (not "NatAlgDiscreteC.py" in student_submission):
        if alg_code_NatAlgDiscrete_dict['A'] != alg_code_NatAlgDiscrete_dict['B']:
            g.write("\n*** error: different algorithm codes in 'NatAlgDiscreteA.py' and 'NatAlgDiscreteB.py'\n")
            error_flag = True
        elif problem_code_NatAlgDiscrete_dict['A'] != problem_code_NatAlgDiscrete_dict['B']:
            g.write("\n*** error: different problem codes in 'NatAlgDiscreteA.py' and 'NatAlgDiscreteB.py'\n")
            error_flag = True
    elif "NatAlgDiscreteA.py" in student_submission and (not "NatAlgDiscreteB.py" in student_submission) and "NatAlgDiscreteC.py" in student_submission:
        if alg_code_NatAlgDiscrete_dict['A'] != alg_code_NatAlgDiscrete_dict['C']:
            g.write("\n*** error: different algorithm codes in 'NatAlgDiscreteA.py' and 'NatAlgDiscreteC.py'\n")
            error_flag = True
        elif problem_code_NatAlgDiscrete_dict['A'] != problem_code_NatAlgDiscrete_dict['C']:
            g.write("\n*** error: different problem codes in 'NatAlgDiscreteA.py' and 'NatAlgDiscreteC.py'\n")
            error_flag = True
    elif (not "NatAlgDiscreteA.py" in student_submission) and "NatAlgDiscreteB.py" in student_submission and "NatAlgDiscreteC.py" in student_submission:
        if alg_code_NatAlgDiscrete_dict['B'] != alg_code_NatAlgDiscrete_dict['C']:
            g.write("\n*** error: different algorithm codes in 'NatAlgDiscreteB.py' and 'NatAlgDiscreteC.py'\n")
            error_flag = True
        elif problem_code_NatAlgDiscrete_dict['B'] != problem_code_NatAlgDiscrete_dict['C']:
            g.write("\n*** error: different problem codes in 'NatAlgDiscreteB.py' and 'NatAlgDiscreteC.py'\n")
            error_flag = True 
            error_flag = True

    g.write("\n")

    # validate witness files

    for witness_digit in ['A', 'B', 'C']:
        path_to_witness = os.path.join(username, "Witness" + witness_digit + ".txt")
        if os.path.exists(path_to_witness):
            if (not "NatAlgDiscrete" + witness_digit +".py" in student_submission):
                g.write("for 'Witness{0}.txt':\n".format(witness_digit))
                g.write("*** error: the file 'Witness{0}.txt' cannot be validated as 'NatAlgDiscrete{0}.py' does not exist\n".format(witness_digit))
                error_flag = True
            else:
                path_to_graph = os.path.join("GraphFiles", problem_code_NatAlgDiscrete_dict[witness_digit] + "Graph" + witness_digit + ".txt")
                if os.path.isfile(path_to_graph):
                    list_of_errors = validate_witness_file(path_to_witness, path_to_graph, tag, tag_length, problem_code_NatAlgDiscrete_dict[witness_digit], alg_code_NatAlgDiscrete_dict[program_digit], graph_digit_NatAlgDiscrete_dict[program_digit], witness_digit, g, parameter_names, parameter_tag, parameter_float)
                    if len(list_of_errors) > 0:
                        error_flag = True
                else:
                    g.write("for 'Witness{0}.txt':\n".format(witness_digit))
                    g.write("*** error: the file 'Witness{0}.txt' cannot be validated as the corresponding graph file '{1}' does not exist\n".format(witness_digit, path_to_graph))
                    error_flag = True
        else:
            g.write("for 'Witness{0}.txt':\n".format(witness_digit))
            g.write("*** warning: the file 'Witness{0}.txt' does not exist\n".format(witness_digit))
            warning_flag = True

    # final validation message

    if error_flag == False and warning_flag == False:
        g.write("\nThe submission has successfully validated!\n")
    elif error_flag == False and warning_flag == True:
        g.write("\nThe submission has successfully validated! However, there were warnings.\n")
    else:
        g.write("\n*** ERRORS: There were fatal errors!\n")

    g.close()




















    
