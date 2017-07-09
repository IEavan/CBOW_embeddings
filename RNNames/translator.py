import unicodedata
import string
import re

import torch
from torch.autograd import Variable

# Constants
PATH = "name_data/eng-fra.txt"
ALLOWED_CHARS = string.ascii_letters + " ,.!?'"
MAX_TRAINING_LENGTH = 10
ALLOWED_PREFIXES = (
        "i am", "i m",
        "you are", "you re",
        "he is", "he s",
        "she is", "she s",
        "they are", "they re",
        "we are", "we re",
        "it is", "it s"
)

# Reseverd Tokens
SOS = 0  # Start of Sentence
EOS = 1  # End of Sentence

# Helper methods
def unicode_to_ascii(s):
    return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != "Mn"
            and c in ALLOWED_CHARS
        )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([^ ])([.,!?])", r"\1 \2", s)
    s = re.sub(r"[^a-z.,!? ]", r" ", s)
    s = re.sub(r" {2,}", r" ", s)
    return s

def read_data(reverse=False):
    with open(PATH, encoding='utf-8') as text:
        lines = text.read().strip().split('\n')

        pairs = []
        for line in lines:
            pair = []
            for sentence in line.split('\t'):
                pair.append(normalize_string(sentence))

            if is_simple(pair):
                pairs.append(pair)
        
        if not reverse:
            input_lang = Lang("English")
            output_lang = Lang("French")
        else:
            pairs = [list(reversed(pair)) for pair in pairs]
            input_lang = Lang("French")
            output_lang = Lang("English")

        return input_lang, output_lang, pairs

def is_simple(pair):
    return len(pair[0].split(" ")) <= MAX_TRAINING_LENGTH and \
            len(pair[1].split(" ")) <= MAX_TRAINING_LENGTH and \
            pair[0].startswith(ALLOWED_PREFIXES)

# Define a class to hold information about a language
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
        
        self.n_words += 1

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

# Read data into lang objects
print("Reading Data...")
input_lang, output_lang, pairs = read_data()
print("{} filtered training pairs loaded".format(len(pairs)))
print("Indexing...")
for pair in pairs:
    input_lang.add_sentence(pair[0])
    output_lang.add_sentence(pair[1])

# print(pairs[:20])
# print(input_lang.word2index["not"])
# print(output_lang.word2index["mais"])

# Define Encoder Network
class Encoder(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super(Encoder, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        
        # Input is one hot vector and can used as proxy for vocabulary
        # Embedding dim is the same size as hidden dim
        self.embeddings = torch.nn.Embedding(input_size, hidden_dims)

        # GRU Input dim is the same size as the hidden dim
        # since the embedding dim is the same as the hidden dim
        self.gru = torch.nn.GRU(hidden_dims, hidden_dims)

    def forward(self, word_var, prev_hidden):
        embed = self.embeddings(word_var)
        output, hidden = self.gru(embed, prev_hidden)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(1, 1, self.hidden)
        return Variable(hidden)

# Define Decoder Network (with attention)
class Decoder(torch.nn.Module):
    def __init__(self, output_dims, hidden_dims, max_length=20):
        super(Decoder, self).__init__()

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.max_length  = max_length

        # Output dims is proxy for output vocabulary
        self.embedding = torch.nn.Embedding(output_dims, hidden_dim)
        self.attention = torch.nn.Linear(hidden_dim * 2, max_length)
        self.attention_combine = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dims, hidden_dims)
        self.out = torch.nn.Linear(hidden_dims, output_dims)

        self.logsoftmax = torch.nn.LogSoftmax()
        self.softmax = torch.nn.softmax()
        self.relu = torch.nn.ReLu()

    def forward(self, prev_word, prev_hidden, encoder_outputs):
        embedded = self.embedding(prev_word).view(1,1,-1)
        attention_input = torch.cat((embedded[0], prev_hidden[0]), 1)
        attention_weights = self.softmax(self.attention(attention_inputs))

        focused_encoder_outputs = torch.bmm(attention_weights.unsqueeze(0),
                                            encoder_outputs.unsqueeze(0))
        gru_input = self.attention_combine(torch.cat((focused_encoder_outputs[0],
                                                      embedded[0]), 1))
        gru_output, hidden = self.gru(gru_input, prev_hidden)
        probabilities = self.logsoftmax(self.out(gru_output))
        return probabilities, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(1,1,self.hidden_dims))
        return hidden
