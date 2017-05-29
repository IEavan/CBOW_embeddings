import torch
from torch.autograd import Variable

# Constants
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
LEARNING_RATE = 1e-1
EPOCHS = 300

# Data
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# Create word and label dictionaries
# To convert words and labels to integers
word_to_id = {}
label_to_id = {}

for sentence, label_list in training_data:
    for word in sentence:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)
    for label in label_list:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

# Define LSTM Model
class LSTM_POS_Tagger(torch.nn.Module):
    def __init__(self, embedding_dims, hidden_dims, vocab_size, label_size):
        super(LSTM_POS_Tagger, self).__init__()

        self.hidden_dims = hidden_dims

        # Create embeddings
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dims)

        # Define the LSTM
        self.lstm = torch.nn.LSTM(embedding_dims, hidden_dims) # in_dims, out_dims

        # Create linear layer
        self.hidden_to_pos_tags = torch.nn.Linear(hidden_dims, label_size)
        
        # Init hidden states
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Create a variable for the initial LSTM hidden state
        hidden = (Variable(torch.zeros(1,1,self.hidden_dims)),
                Variable(torch.zeros(1,1,self.hidden_dims)))
        return hidden

    def forward(self, sentence):
        word_vectors = self.embeddings(sentence)
        hidden_states, _ = self.lstm(word_vectors.view(len(sentence), 1, -1), self.hidden)
        tag_energies = self.hidden_to_pos_tags(hidden_states.view(len(sentence), -1))
        tag_probs = torch.nn.functional.log_softmax(tag_energies)
        return tag_probs

# Helper Functions
def prepare_data(sentence, word_to_id):
    ids = [word_to_id[word] for word in sentence]
    tensor = Variable(torch.LongTensor(ids))
    return tensor

# Training 
model = LSTM_POS_Tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_id), len(label_to_id))
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for i in range(EPOCHS):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        sent_var = prepare_data(sentence, word_to_id)
        tags_var = prepare_data(tags, label_to_id)
        tag_probs = model(sent_var)
        loss = loss_function(tag_probs, tags_var)
        loss.backward()
        optimizer.step()
    if i % 30 is 0:
        print("Current training loss is {:.3}".format(loss.data[0]))

print("Sentence {}".format(training_data[0][0]))
print("has following tag probs: {}".format(model(prepare_data(training_data[0][0], word_to_id))))
