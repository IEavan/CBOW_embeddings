import torch
from torch.autograd import Variable
import torch

# Constants
CONTEXT_SIZE = 2
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processse. In effect,
we conjure the spirits of the compute with out spells.""".split()
EMBEDDING_DIMENSIONS = 4

# Transform raw text into a list of labeled training examples
word_to_index = {word: i for i, word in enumerate(raw_text)}
VOCAB_SIZE = len(raw_text)

data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):

    # Extract the range that we are interested in
    context = raw_text[i-CONTEXT_SIZE:i+CONTEXT_SIZE + 1]

    # Seperate the context words from the label word
    label = context[CONTEXT_SIZE]
    del context[CONTEXT_SIZE]

    # Append the seperate parts to the training data
    data.append((context, label))


class CBOW(torch.nn.Module):

    def __init__(self):
        super(CBOW, self).__init__()
        
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


def make_context_vector(context, word_to_index):
    indicies = [word_to_index[word] for word in context]
    tensor = torch.LongTensor(indicies)
    return Variable(tensor)

# Training setup
model = CBOW()
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
losses = []

# Training loop
for epoch in range(100):
    for training_example in data:
        context = make_context_vector(training_example[0], word_to_index)
        log_probs = model(context)
        target_variable = Variable(torch.LongTensor(
            [word_to_index[training_example[1]]]))
        loss = loss_function(log_probs, target_variable)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.data[0])
    print("Average loss for epoch {} is {:.3}"
            .format(epoch + 1, 1/len(losses) * sum(losses)))

print("First five embeddings are".format(model.embedding[:5]))
print("Vectors for (process) and  (processes) are {} and {}"
        .format(model.embedding[word_to_index["process"]],
            model.embedding[word_to_index["processes"]]))
