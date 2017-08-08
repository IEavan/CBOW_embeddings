from random import random, shuffle
import translator_core as tc
import torch
from torch.autograd import Variable

def load_models_and_data():
    """
    Loads the encoder, decoder, data and lang objects
    returns: encoder, decoder, input_lang, output_lang, training_data_pairs
    """
    input_lang, output_lang, pairs = tc.read_data()
    encoder = tc.Encoder(input_lang.n_words, tc.HIDDEN_DIMS)
    decoder = tc.Decoder(output_lang.n_words, tc.HIDDEN_DIMS, max_length=15)
    tc.load_parameters(encoder, decoder)

    return encoder, decoder, input_lang, output_lang, pairs

def init_train_utils(encoder, decoder, learing_rate=1e-2):
    """
    Creates necessary objects for training
    returns: criterion, encoder_optimizer, decoder_optimizer
    """
    criterion = torch.nn.NLLLoss()
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learing_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learing_rate)
    return criterion, encoder_optimizer, decoder_optimizer

def train_iteration(encoder, decoder,
                    encoder_optimizer, decoder_optimizer,
                    criterion, training_pair, teacher_forcing_probability=0.5):
    """
    Runs a single training iteration with backprop
    returns: average loss per word
    """
    # Clear gradient buffers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Establish tracking variables
    encoder_outputs = Variable(torch.zeros(decoder.max_length, encoder.hidden_dims))
    if tc.USE_CUDA:
        encoder_outputs = encoder_outputs.cuda()
    loss = 0

    # Run Encoder
    encoder_hidden = encoder.init_hidden()
    for i, word in enumerate(training_pair[0]):
        output, encoder_hidden = encoder(word, encoder_hidden)
        encoder_outputs[i] = output[0][0]

    # Run Decoder
    decoder_hidden = encoder_hidden
    start_token = Variable(torch.LongTensor([tc.SOS]))
    if tc.USE_CUDA:
        start_token = start_token.cuda()
    prev_word = start_token

    for i, word in enumerate(training_pair[1]):
        output, decoder_hidden = decoder(prev_word, decoder_hidden, encoder_outputs)
        _, max_index = torch.max(output, 1)
        max_index = max_index.data[0][0]

        # Teacher forcing half the time
        if random() < teacher_forcing_probability:
            prev_word = word
        else:
            prev_word = Variable(torch.LongTensor([max_index]))

        if tc.USE_CUDA:
            prev_word = prev_word.cuda()

        loss += criterion(output, word)

        if max_index is tc.EOS:
            break


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss / (i + 1)

def train_epochs(n_epochs, encoder, decoder,
                 encoder_optimizer, decoder_optimizer,
                 criterion, input_lang, output_lang, pairs,
                 early_termination=None, print_every=100):
    """ Trains the model """

    finished = False
    for epoch in range(n_epochs):
        print("Starting Epoch {}".format(epoch + 1))
        shuffle(pairs)
        losses = []
        for i, pair, in enumerate(pairs):
            training_pair = tc.prepare_pair(pair, input_lang, output_lang)
            loss = train_iteration(encoder, decoder, encoder_optimizer,
                                   decoder_optimizer, criterion,
                                   training_pair, teacher_forcing_probability=0.5)
            losses.append(loss.data[0])

            if (i + 1) % print_every == 0:
                print("Average training loss: {}".format(sum(losses) / print_every))
                losses = []

            if early_termination is not None:
                if i >= early_termination:
                    finished = True
                    break
        print()

        if finished:
            break

if __name__ == "__main__":
    
    # Prompt user for params
    n_epochs = int(input("How many epochs (0 to stop early): "))
    if n_epochs == 0:
        early_termination = int(input("How many iterations: "))
        n_epochs = 1
    else:
        early_termination = None
    print_every = int(input("How often to print: "))
    learing_rate = float(input("Learning rate: "))

    # Load and create relevant objects
    print("\nLoading data...")
    encoder, decoder, input_lang, output_lang, pairs = load_models_and_data()
    criterion, encoder_optimizer, decoder_optimizer = \
        init_train_utils(encoder, decoder, learing_rate=learing_rate,)

    print("Starting training...")
    train_epochs(n_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 criterion, input_lang, output_lang, pairs, early_termination, print_every)

    # Save parameters
    print("Saving...")
    tc.save_parameters(encoder, decoder)
