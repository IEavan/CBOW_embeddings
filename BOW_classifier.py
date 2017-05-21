import torch
from torch.autograd import Variable

# Constants
CONTEXT_SIZE = 2
EMBEDDING_DIMENSIONS = 4
CORPUS_PATH = "Count of Monte Cristo"
EPOCHS = 100
LEARNING_RATE = 3e-2

# Open Count of Monte Cristo
corpus = open(CORPUS_PATH, "r")

# Transform raw text into a list of labeled training examples
word_to_index = {}
data = []
context_window = []

# Iterate over the words in the text corpus
for word in corpus.read(500):

    # If the sliding window isn't full
    # Keep adding words to it
    if len(context_window) < 2 * CONTEXT_SIZE + 1:

        # If a word is not in the map, add it
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

        # Add the word to the sliding window
        context_window.append(word)

    # If the window is big enough, construct training examples
    else:

        # Seperate context from target word
        context = context_window[:CONTEXT_SIZE] + context_window[-CONTEXT_SIZE:]
        target = context_window[CONTEXT_SIZE]

        # Add as training example to data
        data.append((context, target))

        # shift window by one
        del context_window[0]
        context_window.append(word)
        
        # If a word is not in the map, add it
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

# Size of vocabulary is the size of the map
VOCAB_SIZE = len(word_to_index)

# Define continuous bag of words model
class CBOW(torch.nn.Module):

    def __init__(self):
        super(CBOW, self).__init__()
        
        # Create word embedding variables and affine transform variables
        self.embedding = Variable(torch.rand(VOCAB_SIZE, EMBEDDING_DIMENSIONS))
        self.linear = torch.nn.Linear(EMBEDDING_DIMENSIONS, VOCAB_SIZE)

    def forward(self, inputs):

        # Take in tensor of context indicies and transform to embeddings
        embeds = [self.embedding[i.data] for i in inputs]
        
        # Sum the embeddings and apply a linear transform
        sum_embeds = sum(embeds)
        transformed_embeds = self.linear(sum_embeds)

        # Return the log softmax probabilities
        return torch.nn.functional.log_softmax(transformed_embeds)


# Create torch variable from context words
def make_context_vector(context, word_to_index):
    indicies = [word_to_index[word] for word in context]
    tensor = torch.LongTensor(indicies)
    return Variable(tensor)

# Training setup
model = CBOW()
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
losses = []

# Training loop
for epoch in range(EPOCHS):

    # Iterate over the training data in data
    for training_example in data:

        # Extract context as a torch variable
        context = make_context_vector(training_example[0], word_to_index)

        # Compute log_probabilities from the network
        log_probs = model(context)

        # Extract the target word class as a torch variable
        target_variable = Variable(torch.LongTensor(
            [word_to_index[training_example[1]]]))

        # Compute negative log likelyhood loss
        loss = loss_function(log_probs, target_variable)

        # Backprobagate gradients and update variables
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Record loss
        losses.append(loss.data[0])

    # Report traing error
    if epoch % 10 is 0:
        print("Average loss for epoch {} is {:.3}"
                .format(epoch + 1, 1/len(losses) * sum(losses)))

print("First five embeddings are {}".format(model.embedding[:5]))
