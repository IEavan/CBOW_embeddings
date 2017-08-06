import translator_core as tc
import torch
from torch.autograd import Variable

class Model():
    def __init__(self):
        self.input_lang, self.output_lang, _ = tc.read_data()
        self.encoder = tc.Encoder(self.input_lang.n_words, tc.HIDDEN_DIMS)
        self.decoder = tc.Decoder(self.output_lang.n_words,
                                  tc.HIDDEN_DIMS, max_length=15)

        tc.load_parameters(self.encoder, self.decoder)

    def translate(self, sentence):
        """ Runs the provided encoder and decoder to translate a string
        from the input_lang to the output_lang provided that the encoder
        and decoder are trained on that pairovided encoder and decoder
        to translate a string from the input_lang to the output_lang
        provided that the encoder and decoder are trained on that pair"""

        tokenized = tc.normalize_string(sentence)
        input_variable = tc.sentence_to_variable(tokenized, self.input_lang)

        encoder_outputs = Variable(torch.zeros(self.decoder.max_length,
                                               self.encoder.hidden_dims))
        if tc.USE_CUDA:
            encoder_outputs = encoder_outputs.cuda()
        encoder_hidden = self.encoder.init_hidden()

        for i, word in enumerate(input_variable):
            encoder_out, encoder_hidden = self.encoder(word, encoder_hidden)
            encoder_outputs[i] = encoder_out[0][0]

        decoder_hidden = encoder_hidden
        prev_word = Variable(torch.LongTensor([tc.SOS]))
        if tc.USE_CUDA:
            prev_word = prev_word.cuda()

        result = ""

        for i in range(self.decoder.max_length):
            decoder_out, decoder_hidden = self.decoder(prev_word,
                                                       decoder_hidden,
                                                       encoder_outputs)
            _, max_index = torch.max(decoder_out, 1)
            max_index = max_index.data[0][0]

            if max_index is tc.EOS:
                break

            prev_word = Variable(torch.LongTensor([max_index]))
            if tc.USE_CUDA:
                prev_word = prev_word.cuda()
            result += self.output_lang.index2word[max_index] + " "

        return result

if __name__ == "__main__":

    print("Loading model...")
    model = Model()

    print("Model Loaded, enter a phrase to translate, use q to quit")
    print("Input phrases must start with (s)he/you/I/they/it is/are/am")
    print("and be no longer than ten words")

    while True:
        english_text = input("\nInput:  ")

        if english_text is not 'q':
            # Check to make sure english_text is translatable by the model

            english_text = tc.normalize_string(english_text)
            valid = True
            if not tc.is_simple((english_text, "")):
                print("Input is either too long or")
                print("does not start with (s)he/you/I/they/it is/are/am")
                valid = False

            for word in tc.normalize_string(english_text).split(" "):
                if word not in model.input_lang.word2index:
                    print("The word '{}' was not recognized by the model".format(word))
                    print("Please try again")
                    valid = False

            if not valid:
                continue

            print("Output: {}".format(model.translate(english_text)))
        else:
            print("Goodbye")
            break
