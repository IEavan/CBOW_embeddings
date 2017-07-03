# TODO: Import and structure data into training examples
# Define model
# Write training loop
# Write evaluation and testing scripts


import glob
import unicodedata
import string
import random

import torch
from torch.autograd import Variable

# Constants
allowed_chars = string.ascii_letters + " .,;'"
n_chars = len(allowed_chars)
n_hidden = 128
training_iterations = 10000
print_every = 500

# Converts unicode strings into strings containing only ascii
# Characters. Accented letters will have their accents
# removed and other unicode characters will be removed
def unicode_to_ascii(s):
    return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != "Mn"
            and c in allowed_chars
        )

# Returns a list of paths matching a given pattern
def find_files(path):
    return glob.glob(path)

# Opens a file of the given name and
# returns a list of the lines in ascii format
def read_lines(filename):
    with open(filename, encoding="UTF-8") as lines:
        names = lines.read().strip().split('\n')
        return [unicode_to_ascii(name) for name in names]

# Declare data containers
category_lines = {}
categories = []

# Load data into containers
for filename in find_files("name_data/names/*.txt"):
    category = filename.split('/')[-1].split('.')[0]
    categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(categories)

# Helper methods for converting strings to onehot tensors
def letter_to_variable(letter):
    one_hot_tensor = torch.zeros(1, n_chars)
    index = allowed_chars.find(letter)
    one_hot_tensor[0][index] = 1
    return Variable(one_hot_tensor)

def name_to_variable(name):
    one_hot_tensor = torch.zeros(len(name), 1, n_chars)
    # Loop over and add ones at a corresponding index
    for i in range(len(name)):
        index = allowed_chars.find(name[i])
        one_hot_tensor[i][0][index] = 1
    return Variable(one_hot_tensor)


