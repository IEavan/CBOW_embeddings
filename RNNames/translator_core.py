"""
Translation core Module, provides the basical functionality for
using the encoder and decoder and preparing data to be fed
into the encoder and decoder.
"""

from random import shuffle
import unicodedata
import string
import re

import torch
from torch.autograd import Variable

# Constants
USE_CUDA = torch.cuda.is_available()
HIDDEN_DIMS = 256
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
ENCODER_PATH = "model_params/encoder"
DECODER_PATH = "model_params/decoder"

# Reseverd Tokens
SOS = 0  # Start of Sentence
EOS = 1  # End of Sentence

# Helper methods
def unicode_to_ascii(s):
    """
    Converts a unicode string to an ascii string
    """
    return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != "Mn"
            and c in ALLOWED_CHARS
        )

def normalize_string(s):
    """
    Prepares a string to be ready to be split into tokens.

    Takes in a unicode string and outputs an ascii string
    that non-standard punctuation removed and spaces
    are inserted between punctuation marks.
    i.e. I'm --> I m
         By. --> By .
    """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([^ ])([.,!?])", r"\1 \2", s)
    s = re.sub(r"[^a-z.,!? ]", r" ", s)
    s = re.sub(r" {2,}", r" ", s)
    return s

def read_data(path=PATH, reverse=False):
    """
    Reads the translation text file and outputs two language objects
    and a list of training pairs shuffled in a random order
    """
    with open(path, encoding='utf-8') as text:
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

        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        shuffle(pairs)

        return input_lang, output_lang, pairs

def is_simple(pair):
    """
    A test to determine whether or not a training pair conforms
    to certain length and vocab restrictions to enforce simplicity
    """
    return len(pair[0].split(" ")) <= MAX_TRAINING_LENGTH and \
            len(pair[1].split(" ")) <= MAX_TRAINING_LENGTH and \
            pair[0].startswith(ALLOWED_PREFIXES)

# Define a class to hold information about a language
class Lang:
    """
    Stores data about the vocabulary of a language,
    such as converting a word to an index.
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_word(self, word):
        """
        Method for adding a string to the lang dictionaries
        """
        if not word in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1


    def add_sentence(self, sentence):
        """
        Calls add_word on each word in a sentence string
        """
        for word in sentence.split(" "):
            self.add_word(word)

# Define Encoder Network
class Encoder(torch.nn.Module):
    """
    Encoder class provides functionality for converting a sequence
    of word indicies in the input language to a vector to be sent to be decoded
    """
    def __init__(self, input_size, hidden_dims):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dims = hidden_dims

        # Input size is the size of the input vocabulary
        # Embedding dim is the same size as hidden dim
        self.embeddings = torch.nn.Embedding(input_size, hidden_dims)

        # GRU Input dim is the same size as the hidden dim
        # since the embedding dim is the same as the hidden dim
        self.gru = torch.nn.GRU(hidden_dims, hidden_dims)

    def forward(self, word_var, prev_hidden):
        """
        Single forward pass of the recurrent model
        Model contains one embedding layer and one gated recurrent unit (GRU)
        """
        embed = self.embeddings(word_var).view(1, 1, -1)
        output, hidden = self.gru(embed, prev_hidden)
        return output, hidden

    def init_hidden(self):
        """
        Return a vector of the correct dimensions
        to be fed as the first hidden state of the encoder
        """
        hidden = torch.zeros(1, 1, self.hidden_dims)
        hidden = Variable(hidden)
        if USE_CUDA:
            return hidden.cuda()
        return hidden

# Define Decoder Network (with attention)
class Decoder(torch.nn.Module):
    """
    Decoder takes the output of the encoder and outputs predictions for
    the words in the output language
    """
    def __init__(self, output_dims, hidden_dims, max_length=20):
        super(Decoder, self).__init__()

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.max_length = max_length

        # Output dims is proxy for output vocabulary
        self.embedding = torch.nn.Embedding(output_dims, hidden_dims)
        self.attention = torch.nn.Linear(hidden_dims * 2, max_length)
        self.attention_combine = torch.nn.Linear(hidden_dims * 2, hidden_dims)
        self.gru = torch.nn.GRU(hidden_dims, hidden_dims)
        self.out = torch.nn.Linear(hidden_dims, output_dims)

        self.logsoftmax = torch.nn.LogSoftmax()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(self, prev_word, prev_hidden, encoder_outputs):
        """
        Single forward pass of the recurrent decoder.

        Inputs: previous word LongTensor word index
                prev_hidden state output
                    or the encoder hidden state on the first pass
                individual outputs for each iteration of the encoder
                    as a single tensor

        Outputs: Log probabilities for each word in the output language
                 hidden state to be passed into the next iteration
        """
        embedded = self.embedding(prev_word).view(1, 1, -1)
        attention_input = torch.cat((embedded[0], prev_hidden[0]), 1)
        attention_weights = self.softmax(self.attention(attention_input))

        focused_encoder_outputs = torch.bmm(attention_weights.unsqueeze(0),
                                            encoder_outputs.unsqueeze(0))
        gru_input = self.attention_combine(torch.cat((focused_encoder_outputs[0],
                                                      embedded[0]), 1)).unsqueeze(0)
        gru_input = self.relu(gru_input)
        gru_output, hidden = self.gru(gru_input, prev_hidden)
        probabilities = self.logsoftmax(self.out(gru_output[0]))
        return probabilities, hidden

    def init_hidden(self):
        """
        Returns a vector of zeros to be fed into forward
        function on the first iteration if no output from the encoder exists
        """
        hidden = Variable(torch.zeros(1, 1, self.hidden_dims))
        if USE_CUDA:
            return hidden.cuda()
        return hidden

# Data Preperation helper functions
def sentence_to_indicies(sentence, lang):
    """ Uses the lang object to convert a string into a list of indicies """
    return [lang.word2index[word] for word in sentence.split(" ")]

def sentence_to_variable(sentence, lang):
    """
    Takes a string and converts it into a LongTensor of indicies
    with cuda support and appends he EOS token
    """
    indicies = sentence_to_indicies(sentence, lang)
    indicies.append(EOS)
    variable = Variable(torch.LongTensor(indicies))
    if USE_CUDA:
        return variable.cuda()
    return variable

def prepare_pair(pair, input_lang, output_lang):
    """Takes a input output pair and prepares it for the encoder and decoder"""
    return (sentence_to_variable(pair[0], input_lang),
            sentence_to_variable(pair[1], output_lang))

def save_parameters(encoder, decoder, enc_path=ENCODER_PATH, dec_path=DECODER_PATH):
    """ Saves the model parameters in the specified path """
    torch.save(encoder.state_dict(), enc_path)
    torch.save(decoder.state_dict(), dec_path)

def load_parameters(encoder, decoder, enc_path=ENCODER_PATH, dec_path=DECODER_PATH):
    """ Loads the model parameters from the specified path """
    try:
        encoder.load_state_dict(torch.load(enc_path))
        decoder.load_state_dict(torch.load(dec_path))
    except FileNotFoundError:
        print("No parameters where found")
