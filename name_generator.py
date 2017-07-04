# TODO: 
# Write training loop
# Write evaluation and testing scripts


import glob
import unicodedata
import string
import random
from tqdm import *

import torch
from torch.autograd import Variable

# Constants
allowed_chars = string.ascii_letters + " .,;'"
n_chars = len(allowed_chars)
n_hidden = 128
learning_rate = 0.0005
training_iterations = 10000
print_every = 1000

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

def name_to_target(name):
    name = name + " "
    indicies = [allowed_chars.find(char) for char in name]
    target = Variable(torch.LongTensor(indicies))
    return target

def category_to_variable(category):
    one_hot_tensor = torch.zeros(1, n_categories)
    
    if type(category) is int:
        one_hot_tensor[0][category_index] = 1
    elif type(category) is str:
        index = categories.index(category)
        one_hot_tensor[0][index] = 1
    
    return Variable(one_hot_tensor)


# Define Generative Model
class Generator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_probability=0.1):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size

        # parameters
        self.c2i = torch.nn.Linear(input_size + hidden_size + n_categories, output_size)
        self.c2h = torch.nn.Linear(input_size + hidden_size + n_categories, hidden_size)
        self.i2o = torch.nn.Linear(output_size + hidden_size, output_size)
        # Read c2i as combined value to intermediate value;
        #      c2h as combined value to hidden value
        #      i2o as intermediate value to output value

        # Define internal functions
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = torch.nn.Dropout(drop_probability)

    def forward(self, category, prev_out, prev_hidden):
        # Function to combute a single iteration of the recurrent generator
        # Outputting negative log probabilities for a letter and the next hidden state
        combined = torch.cat((category, prev_out, prev_hidden), 1)
        intermediate = self.c2i(combined)
        hidden = self.c2h(combined)
        
        combined_intermediate = torch.cat((intermediate, hidden), 1)
        output = self.i2o(combined_intermediate)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        # Function to generate a hidden value with the correct dimensions
        return Variable(torch.zeros(1, self.hidden_size))

# Training
model = Generator(n_chars, n_hidden, n_chars)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_random_example():
   rand_category_index = random.randint(0, n_categories - 1)
   category = categories[rand_category_index]
   rand_line_index = random.randint(0, len(category) - 1)
   line = category_lines[category][rand_line_index]
   return line, category, rand_category_index

loss_accumulator = 0
for iteration in tqdm(range(training_iterations)):
    # Get training example
    line, category, category_index = get_random_example()

    # Convert to torch.autograd variables
    name_variable = name_to_variable(line)
    category_variable = category_to_variable(category_index)
    target = name_to_target(line)

    # Append special start char to name
    name_variable = torch.cat(
            (
                torch.unsqueeze(letter_to_variable(";"),0),
                name_variable
            ),0)

    # Init a new blank hidden state for the model
    hidden_state = model.init_hidden()

    # Iteratate over the name, accumulating loss
    loss = 0
    for i in range(len(target)):
        out, hidden_state = model(category_variable,
                                  name_variable[i],
                                  hidden_state)
        loss += criterion(out, target[i])

    loss_accumulator += loss.data[0]

    # Backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Print averaged loss
    if (iteration + 1) % print_every is 0:
        print("Current Training Loss is {}"
                .format(loss_accumulator / print_every))
        loss_accumulator = 0

# Test Generation
def generate(category):
    # Create start token
    prev_char = letter_to_variable(";")

    # Initialize a new hidden state
    hidden_state = model.init_hidden()

    # Transform inputs into torch autograd variables
    category = category_to_variable(category)

    # Iterate over results from model
    result = ""
    counter = 0
    while True:
        out, hidden_state = model(category, prev_char, hidden_state)
        
        char_index = torch.max(out, 1)[1].data[0][0]
        char = allowed_chars[char_index]
        prev_char = letter_to_variable(char)

        if char is " " or counter > 20:
            break
        else:
            result += char
            counter += 1

    return result

print("Russian == {}".format(generate("Russian")))
print("Arabic == {}".format(generate("Arabic")))
print("Spanish == {}".format(generate("Spanish")))
print("Polish == {}".format(generate("Polish")))
print("German == {}".format(generate("German")))
