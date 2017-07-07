import glob
import unicodedata
import string
import random

from torch import zeros, LongTensor
from torch import max as tmax
from torch.autograd import Variable

# Required Constants
ALLOWED_CHARS = string.ascii_letters + " .,;'"
N_CHARS = len(ALLOWED_CHARS)

# Converts unicode strings into strings containing only ascii
# Characters. Accented letters will have their accents
# removed and other unicode characters will be removed
def unicode_to_ascii(s):
    return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != "Mn"
            and c in ALLOWED_CHARS
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

# Load the data from the name data directory into
# A dictionary of names and a list of languages
category_lines = {}
categories = []

for filename in find_files("name_data/names/*.txt"):
    category = filename.split('/')[-1].split('.')[0]
    categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

# Shorthand for number of categories
N_CATEGORIES = len(categories)

# Return the data containers
def load_data():
    return category_lines, categories


# Helper methods for converting strings to onehot tensors
def letter_to_variable(letter):
    one_hot_tensor = zeros(1, N_CHARS)
    index = ALLOWED_CHARS.find(letter)
    one_hot_tensor[0][index] = 1
    return Variable(one_hot_tensor)

def name_to_variable(name):
    one_hot_tensor = zeros(len(name), 1, N_CHARS)
    # Loop over and add ones at a corresponding index
    for i in range(len(name)):
        index = ALLOWED_CHARS.find(name[i])
        one_hot_tensor[i][0][index] = 1
    return Variable(one_hot_tensor)

# Converts a string name into an autograd Variable
# To be used as a target in NLLLoss()
def name_to_target(name):
    name = name + " "
    indicies = [ALLOWED_CHARS.find(char) for char in name]
    target = Variable(LongTensor(indicies))
    return target

# Converts a category into an autograd Variable
# Encoded as a one hot tensor
def category_to_variable(category):
    one_hot_tensor = zeros(1, N_CATEGORIES)
    
    if type(category) is int:
        one_hot_tensor[0][category] = 1
    elif type(category) is str:
        index = categories.index(category)
        one_hot_tensor[0][index] = 1
    
    return Variable(one_hot_tensor)

# Converts Log probabilities into a category
def probs_to_category(probs):
    probability, index = tmax(probs, 1)
    category = categories[index.data[0][0]]
    return category, index.data[0][0]

# Extracts a random sample from the dataset
def get_random_example():
   rand_category_index = random.randint(0, N_CATEGORIES - 1)
   category = categories[rand_category_index]
   rand_line_index = random.randint(0, len(category) - 1)
   line = category_lines[category][rand_line_index]
   return line, category, rand_category_index
