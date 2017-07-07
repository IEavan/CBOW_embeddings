import name_utils as utils
import torch
from torch.autograd import Variable

# Constants
N_HIDDEN = 128
TRAINING_ITERATIONS = 5000
PRINT_EVERY = 500
LEARNING_RATE = 0.0005


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
model = RNN(utils.N_CHARS, N_HIDDEN, utils.n_categories)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(line, category, category_index):
    # Convert name to torch variable
    line = utils.name_to_variable(line)
    
    # Reset gradient buffers and initialize the model hidden state
    optimizer.zero_grad()
    hidden = model.init_hidden()

    # Loop over all characters in the name
    # Passing the updated hidden states through
    for i in range(line.size()[0]):
        out, hidden = model(line[i], hidden)

    # Convert negative log probabilities to a category
    predicted_category, p_cat_index = utils.probs_to_category(out)

    # Calculate cross entropy loss and backpropagate
    # Use gradients to update parameters according to Adam optimizer
    loss = criterion(out, Variable(torch.LongTensor([category_index]))) 
    loss.backward()
    optimizer.step()

    return predicted_category, loss.data[0]

def predict(name):
    name_var = utils.name_to_variable(name)
    hidden = model.init_hidden()

    for i in range(name_var.size()[0]):
        out, hidden = model(name_var[i], hidden)

    predicted_category, _ = utils.probs_to_category(out)
    return predicted_category

# Training loop
for iterations in range(TRAINING_ITERATIONS):
    error_accumulator = 0
    line, category, category_index = utils.get_random_example()
    predicted, example_loss = train(line, category, category_index)
    error_accumulator += example_loss

    if iterations % PRINT_EVERY is 0:
        print("Current average training error is {}".format(error_accumulator / 100))
        error_accumulator = 0
        print("{}/{} predicted as {}\n".format(category, line, predicted))

# Example test case
print("Eavan predicted as {}".format(predict("Eavan")))
