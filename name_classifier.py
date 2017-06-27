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

def probs_to_category(probs):
    probability, index = torch.max(probs, 1)
    category = categories[index.data[0][0]]
    return category, index.data[0][0]

#
# Define RNN Model
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # parameters
        self.c2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.c2o = torch.nn.Linear(input_size + hidden_size, output_size)

        self.softmax = torch.nn.LogSoftmax()

    def forward(self, letter, hidden):
       combinded = torch.cat((letter, hidden), 1) 
       
       next_hidden = self.c2h(combinded)
       output = self.c2o(combinded)
       output = self.softmax(output)
       return output, next_hidden

    def init_hidden(self):
       return Variable(torch.zeros(1, self.hidden_size))


# Model initialization
model = RNN(n_chars, n_hidden, n_categories)


def get_random_example():
   rand_category_index = random.randint(0, n_categories - 1)
   category = categories[rand_category_index]
   rand_line_index = random.randint(0, len(category) - 1)
   line = category_lines[category][rand_line_index]
   return line, category, rand_category_index

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train(line, category, category_index):
    # Convert name to torch variable
    line = name_to_variable(line)
    
    # Reset gradient buffers and initialize the model hidden state
    optimizer.zero_grad()
    hidden = model.init_hidden()

    # Loop over all characters in the name
    # Passing the updated hidden states through
    for i in range(line.size()[0]):
        out, hidden = model(line[i], hidden)

    # Convert negative log probabilities to a category
    predicted_category, p_cat_index = probs_to_category(out)

    # Calculate cross entropy loss and backpropagate
    # Use gradients to update parameters according to Adam optimizer
    loss = criterion(out, Variable(torch.LongTensor([category_index]))) 
    loss.backward()
    optimizer.step()

    return predicted_category, loss.data[0]

def predict(name):
    name_var = name_to_variable(name)
    hidden = model.init_hidden()

    for i in range(name_var.size()[0]):
        out, hidden = model(name_var[i], hidden)

    predicted_category, _ = probs_to_category(out)
    return predicted_category

# Training loop
for iterations in range(training_iterations):
    error_accumulator = 0
    line, category, category_index = get_random_example()
    predicted, example_loss = train(line, category, category_index)
    error_accumulator += example_loss

    if iterations % print_every is 0:
        print("Current average training error is {}".format(error_accumulator / 100))
        error_accumulator = 0
        print("{}/{} predicted as {}\n".format(category, line, predicted))

# Example test case
print("Eavan predicted as {}".format(predict("Eavan")))
