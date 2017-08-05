from random import shuffle, random
import unicodedata
import string
import re

import torch
from torch.autograd import Variable

# Constants
USE_CUDA = torch.cuda.is_available()
HIDDEN_DIMS = 256
EPOCHS = 1
PRINT_EVERY = 100
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
USE_SAVING = True
ENCODER_PATH = "model_params/encoder"
DECODER_PATH = "model_params/decoder"
LEARNING_RATE = 1e-2
BREAK_ITER = 100000  # Use to stop after a certain number of iters in an epoch

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
            self.n_words += 1
        else:
            self.word2count[word] += 1


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

shuffle(pairs)


# Define Encoder Network
class Encoder(torch.nn.Module):
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
        embed = self.embeddings(word_var).view(1,1,-1)
        output, hidden = self.gru(embed, prev_hidden)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(1, 1, self.hidden_dims)
        hidden = Variable(hidden)
        if USE_CUDA:
            return hidden.cuda()
        return hidden

# Define Decoder Network (with attention)
class Decoder(torch.nn.Module):
    def __init__(self, output_dims, hidden_dims, max_length=20):
        super(Decoder, self).__init__()

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.max_length  = max_length

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
        embedded = self.embedding(prev_word).view(1,1,-1)
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
        hidden = Variable(torch.zeros(1,1,self.hidden_dims))
        if USE_CUDA:
            return hidden.cuda()
        return hidden

# Data Preperation helper functions
def sentence_to_indicies(sentence, lang):
    return [lang.word2index[word] for word in sentence.split(" ")]

def sentence_to_variable(sentence, lang):
    indicies = sentence_to_indicies(sentence, lang)
    indicies.append(EOS)
    variable = Variable(torch.LongTensor(indicies))
    if USE_CUDA:
        return variable.cuda()
    return variable

def prepare_pair(pair, input_lang, output_lang):
    return (sentence_to_variable(pair[0], input_lang),
            sentence_to_variable(pair[1], output_lang))

def save_parameters(encoder, decoder, enc_path=ENCODER_PATH, dec_path=DECODER_PATH):
    torch.save(encoder.state_dict(), enc_path)
    torch.save(decoder.state_dict(), dec_path)

# Train Model
def train(encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, training_pair):

    # Clear gradient buffers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Establish tracking variables
    encoder_outputs = Variable(torch.zeros(decoder.max_length, encoder.hidden_dims))
    if USE_CUDA:
        encoder_outputs = encoder_outputs.cuda()
    loss = 0

    # Run Encoder
    encoder_hidden = encoder.init_hidden()
    for i, word in enumerate(training_pair[0]):
        output, encoder_hidden = encoder(word, encoder_hidden)
        encoder_outputs[i] = output[0][0]

    # Run Decoder
    decoder_hidden = encoder_hidden
    start_token = Variable(torch.LongTensor([SOS]))
    if USE_CUDA:
        start_token = start_token.cuda()
    prev_word = start_token

    for i, word in enumerate(training_pair[1]):
        output, decoder_hidden = decoder(prev_word, decoder_hidden, encoder_outputs)
        _, max_index = torch.max(output, 1)
        max_index = max_index.data[0][0]

        # Teacher forcing half the time
        if random() > 0.5:
            prev_word = word
        else:
            prev_word = Variable(torch.LongTensor([max_index]))

        if USE_CUDA:
            prev_word = prev_word.cuda()

        loss += criterion(output, word)

        if max_index is EOS:
            break


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss / (i + 1)


# Training Loop
encoder = Encoder(input_lang.n_words, HIDDEN_DIMS)
decoder = Decoder(output_lang.n_words, HIDDEN_DIMS, max_length=15)
criterion = torch.nn.NLLLoss()
encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=LEARNING_RATE)

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

if USE_SAVING:
    try:
        encoder.load_state_dict(torch.load(ENCODER_PATH))
        decoder.load_state_dict(torch.load(DECODER_PATH))
        print("Model parameters found and loaded")
    except FileNotFoundError:
        print("No parameters where found")

print("Starting training with {} epochs".format(EPOCHS))
for epoch in range(EPOCHS):
    print("Epoch {}\n\n".format(epoch + 1))
    epoch_losses = []
    reporting_losses = 0
    for i, pair in enumerate(pairs):
        training_pair = prepare_pair(pair, input_lang, output_lang)
        loss = train(encoder, decoder,
                     encoder_optimizer, decoder_optimizer,
                     criterion, training_pair)

        reporting_losses += loss.data[0]
        epoch_losses.append(loss.data[0])

        if (i + 1) % PRINT_EVERY is 0:
            print("Training loss is {}".format(reporting_losses / PRINT_EVERY))
            reporting_losses = 0

        if (i + 1) % BREAK_ITER is 0:
            print("Training Stopped!")
            break

if USE_SAVING:
    print("Saving parameters")
    save_parameters(encoder, decoder)

def translate(sentence, encoder, decoder, input_lang, output_lang):
    """ Runs the provided encoder and decoder to translate a string
    from the input_lang to the output_lang provided that the encoder
    and decoder are trained on that pairovided encoder and decoder to translate a string
    from the input_lang to the output_lang provided that the encoder
    and decoder are trained on that pair"""
    tokenized = normalize_string(sentence)
    input_variable = sentence_to_variable(tokenized, input_lang)

    encoder_outputs = Variable(torch.zeros(decoder.max_length, encoder.hidden_dims))
    if USE_CUDA:
        encoder_outputs = encoder_outputs.cuda()
    encoder_hidden = encoder.init_hidden()

    for i, word in enumerate(input_variable):
        encoder_out, encoder_hidden = encoder(word, encoder_hidden)
        encoder_outputs[i] = encoder_out[0][0]

    decoder_hidden = encoder_hidden
    prev_word = Variable(torch.LongTensor([SOS]))
    if USE_CUDA:
        prev_word = prev_word.cuda()

    result = ""

    for i in range(decoder.max_length):
        decoder_out, decoder_hidden = decoder(prev_word, decoder_hidden, encoder_outputs)
        _, max_index = torch.max(decoder_out, 1)
        max_index = max_index.data[0][0]

        if max_index is EOS:
            break

        prev_word = Variable(torch.LongTensor([max_index]))
        if USE_CUDA:
            prev_word = prev_word.cuda()
        result += output_lang.index2word[max_index] + " "

    return result

print("IN: I am very cold")
print("OUT: {}".format(translate("I am very cold", encoder, decoder,
                                 input_lang, output_lang)))
print("IN: She is hungry")
print("OUT: {}".format(translate("She is hungry", encoder, decoder,
                                 input_lang, output_lang)))
print("IN: I am stopping")
print("OUT: {}".format(translate("I am stopping", encoder, decoder,
                                 input_lang, output_lang)))
print("IN: He can not control himself")
print("OUT: {}".format(translate("He can not control himself", encoder, decoder,
                                 input_lang, output_lang)))
print("IN: It is falling apart")
print("OUT: {}".format(translate("It is falling apart", encoder, decoder,
                                 input_lang, output_lang)))
