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

        # Char level representation
        lstm_char_dims = 4
        self.char_lstm = torch.nn.LSTM(1, lstm_char_dims)

        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dims) # Create embeddings
        self.lstm = torch.nn.LSTM(embedding_dims + lstm_char_dims, hidden_dims) # Create LSTM | in_dims, out_dims
        self.hidden_to_pos_tags = torch.nn.Linear(hidden_dims, label_size) # Create linear layer
        self.hidden1 = self.init_hidden(lstm_char_dims) # Init hidden states
        self.hidden2 = self.init_hidden(hidden_dims) # Init hidden states

    def init_hidden(self, dims):
        # Create a variable for the initial LSTM hidden state
        hidden = (Variable(torch.zeros(1,1,dims)),
                Variable(torch.zeros(1,1,dims)))
        return hidden

    def forward(self, sentence, chars):
        word_vectors = self.embeddings(sentence)

        char_rep = []
        for i in range(len(chars)):
            letters = Variable(torch.FloatTensor(chars[i])).view(-1,1,1)
            _, rep = self.char_lstm(letters, self.hidden1)
            char_rep.append(rep[0])
        char_rep = torch.stack(char_rep, dim=0)

        hidden_states, _ = self.lstm(
            torch.cat(
                [word_vectors.view(len(sentence), 1, -1), char_rep.view(len(sentence), 1, -1)], 2),
            self.hidden2)
        tag_energies = self.hidden_to_pos_tags(hidden_states.view(len(sentence), -1))
        tag_probs = torch.nn.functional.log_softmax(tag_energies)
        return tag_probs

# Helper Functions
def prepare_data(sentence, word_to_id):
    
    ids = [word_to_id[word] for word in sentence]
    id_tensor = Variable(torch.LongTensor(ids))

    character_nums = [[ord(c) - ord('a') for c in word] for word in sentence]
    return id_tensor, character_nums

# Training 
model = LSTM_POS_Tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_id), len(label_to_id))
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for i in range(EPOCHS):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden1 = model.init_hidden(4)
        model.hidden2 = model.init_hidden(HIDDEN_DIM)
        sent_var, char_list = prepare_data(sentence, word_to_id)
        tags_var, _ = prepare_data(tags, label_to_id)
        tag_probs = model(sent_var, char_list)
        loss = loss_function(tag_probs, tags_var)
        loss.backward()
        optimizer.step()
    if i % 30 is 0:
        print("Current training loss is {:.3}".format(loss.data[0]))

# Show example
print("Sentence {}".format(training_data[0][0]))
print("has following tag probs: {}".format(model(*prepare_data(training_data[0][0], word_to_id))))
