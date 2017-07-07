import name_utils as utils
from tqdm import *

import torch
from torch.autograd import Variable

# Constants
N_HIDDEN = 128
LEARNING_RATE = 0.0005
TRAINING_ITERATIONS = 10000
PRINT_EVERY = 1000


# Define Generative Model
class Generator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_probability=0.1):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size

        # parameters
        self.c2i = torch.nn.Linear(input_size + hidden_size + utils.N_CATEGORIES, output_size)
        self.c2h = torch.nn.Linear(input_size + hidden_size + utils.N_CATEGORIES, hidden_size)
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
model = Generator(utils.N_CHARS, N_HIDDEN, utils.N_CHARS)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


loss_accumulator = 0
for iteration in tqdm(range(TRAINING_ITERATIONS)):
    # Get training example
    line, category, category_index = utils.get_random_example()

    # Convert to torch.autograd variables
    name_variable = utils.name_to_variable(line)
    category_variable = utils.category_to_variable(category_index)
    target = utils.name_to_target(line)

    # Append special start char to name
    name_variable = torch.cat(
            (
                torch.unsqueeze(utils.letter_to_variable(";"),0),
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
    if (iteration + 1) % PRINT_EVERY is 0:
        print("Current Training Loss is {}"
                .format(loss_accumulator / PRINT_EVERY))
        loss_accumulator = 0

# Test Generation
def generate(category):
    # Create start token
    prev_char = utils.letter_to_variable(";")

    # Initialize a new hidden state
    hidden_state = model.init_hidden()

    # Transform inputs into torch autograd variables
    category = utils.category_to_variable(category)

    # Iterate over results from model
    result = ""
    counter = 0
    while True:
        out, hidden_state = model(category, prev_char, hidden_state)
        
        char_index = torch.max(out, 1)[1].data[0][0]
        char = utils.ALLOWED_CHARS[char_index]
        prev_char = utils.letter_to_variable(char)

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
