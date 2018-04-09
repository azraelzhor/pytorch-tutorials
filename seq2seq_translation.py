import unicodedata
import re
import torch
from torch.autograd import Variable

class Lang:
    def __init__(self, lang):
        self.lang = lang
        self.w2i = {"SOS": 0, "EOS": 1}
        self.w2c = {}
        self.i2w = {0: "SOS", 1: "EOS"}
        self.vocab_size = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if word not in self.w2i:
                self.i2w[self.vocab_size] = word
                self.w2i[word] = self.vocab_size
                self.w2c[word] = 1
                self.vocab_size += 1
            else:
                self.w2c[word] += 1

# turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")
    lines = open('data/%s-%s.txt' %(lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

# prepare data
def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words:")
    print(input_lang.lang, input_lang.vocab_size)
    print(output_lang.lang, output_lang.vocab_size)
    return input_lang, output_lang, pairs

def make_tensor(sentence, lang):
    return Variable(torch.LongTensor([lang.w2i[word] for word in sentence.split(' ')]))

# create model
class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        '''
        input_size: vocabulary V of embeddings
        hidden_size: also embeddings dim
        '''
        super(Encoder, self).__init__()
        self.H = hidden_size
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.rnn = torch.nn.GRU(hidden_size, hidden_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        self.hidden = Variable(torch.zeros(1, 1, self.H))

    def forward(self, input):
        embeds = self.word_embeddings(input).view(len(input), 1, -1)
        gru_out, self.hidden = self.rnn(embeds, self.hidden)
        return gru_out

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        '''
        input_size: vocabulary V of embeddings
        hidden_size: also embeddings dim
        '''
        super(Decoder, self).__init__()
        self.H = hidden_size
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        z = self.word_embeddings(input).view(len(input), 1, -1)
        z_relu = self.relu(z)
        gru_out, self.hidden = self.gru(z_relu, self.hidden)
        out = self.linear(gru_out.view(-1, self.H))
        out = self.softmax(out)
        return out

# train model
NUM_EPOCH = 10000
LEARNING_RATE = 0.008
EMBEDDING_DIM = 6
HIDDEN_SIZE = 10
MAX_LENGTH = 5

BEGIN_TOKEN = "SOS"
END_TOKEN = "EOS"

input_lang, output_lang, pairs = prepare_data('sim', 'la', reverse=False)

def train_without_attention():

    encoder = Encoder(input_lang.vocab_size, HIDDEN_SIZE)
    decoder = Decoder(output_lang.vocab_size, HIDDEN_SIZE)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCH):
        total_loss = Variable(torch.Tensor([0]))
        for input_sent, output_sent in pairs:
            target_sent = output_sent + ' ' + END_TOKEN

            optimizer.zero_grad()

            encoder_input_tensor = make_tensor(input_sent, input_lang)
            target_tensor = make_tensor(target_sent, output_lang)

            encoder.init_hidden()
            encoder(encoder_input_tensor)

            decoder_input_tensor = make_tensor(BEGIN_TOKEN, output_lang)

            decoder.hidden = encoder.hidden
            decoded_outputs = []
            for i in range(len(target_tensor)):
                output_tensor = decoder(decoder_input_tensor)
                decoded_outputs.append(output_tensor)
                _, index = output_tensor.data[0].topk(1)
                decoder_input_tensor = Variable(index)

            decoded_outputs = torch.cat(decoded_outputs)

            loss = criterion(decoded_outputs, target_tensor)
            loss.backward()

            total_loss += loss

            optimizer.step()

        print(total_loss.data[0])

class AttentionDecoder(torch.nn.Module):
    def __init__(self,vocab_size, hidden_size, max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.L = max_length
        self.H = hidden_size

        self.word_embeddings = torch.nn.Embedding(vocab_size, self.H)
        self.softmax = torch.nn.Softmax(dim=1)
        # self.alignment = torch.nn.Linear((L + 1) * H, L)
        self.alignment = torch.nn.Linear(self.H, self.L)
        self.combine = torch.nn.Linear(2 * self.H, self.H)
        self.gru = torch.nn.GRU(self.H, self.H)
        self.out = torch.nn.Linear(self.H, vocab_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoded_outputs):
        '''
        Args
        input: index
        hidden: (s_i-1, 1 x 1 x H) 
        '''
        # 1 x 1 x H
        embeds = self.word_embeddings(input).view(1, 1, -1)

        # (S + 1) x 1 x H 
        hidden_combined = torch.cat([hidden, encoded_outputs], dim=0)

        # (S + 1) x 1 x L
        weights = self.alignment(hidden_combined)

        # 1 x L
        weights = weights.mean(dim=0)

        # 1 x L
        weights_softmax = self.softmax(weights)
        
        # 1 x S => repeat => H x 1 x S => transpose => S x 1 x H => * => S x 1 x H => sum(dim=0, keepdim=True) => 1 x 1 x H
        S = encoded_outputs.size()[0]
        context = (weights_softmax[:, :S].repeat(self.H, 1, 1).transpose(2, 0) * encoded_outputs).sum(dim=0, keepdim=True)

        # 1 x 1 x H cat 1 x 1 x H => 1 x (H + H)
        input_context_combined = torch.cat([embeds, context], dim=2).view(1, -1)

        # 1 x H => unsqueeze => 1 x 1 x H
        gru_in = self.combine(input_context_combined).unsqueeze(0)

        # 1 x 1 x H, 1 x 1 x H
        gru_out, gru_hidden = self.gru(gru_in, hidden)

        out = self.out(gru_out.view(-1, self.H))
        out = self.log_softmax(out)

        return out, gru_hidden

def train_with_attention():

    encoder = Encoder(input_lang.vocab_size, HIDDEN_SIZE)
    attention_decoder = AttentionDecoder(output_lang.vocab_size, HIDDEN_SIZE)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(attention_decoder.parameters()), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCH):
        total_loss = Variable(torch.Tensor([0]))
        for input_sent, output_sent in pairs:
            target_sent = output_sent + ' ' + END_TOKEN
            optimizer.zero_grad()

            encoder_input_tensor = make_tensor(input_sent, input_lang)
            target_tensor = make_tensor(target_sent, output_lang)

            encoder.init_hidden()
            encoded_outputs = encoder(encoder_input_tensor)

            decoder_input_tensor = make_tensor(BEGIN_TOKEN, output_lang)

            decoder_hidden = encoder.hidden
            output_tensor = decoder_input_tensor
            decoded_outputs = []
            for i in range(len(target_tensor)):
                output_tensor, decoder_hidden = attention_decoder(decoder_input_tensor, decoder_hidden, encoded_outputs)
                decoded_outputs.append(output_tensor)
                _, index = output_tensor.data[0].topk(1)
                decoder_input_tensor = Variable(index)

            decoded_outputs = torch.cat(decoded_outputs)

            loss = criterion(decoded_outputs, target_tensor)
            loss.backward()

            total_loss += loss

            optimizer.step()

        print('Epoch %d, loss = %f' % (epoch + 1, total_loss.data[0]))

    print(evaluate(encoder, attention_decoder, "no way"))

def evaluate(encoder, attention_decoder, sentence):
    encoder_input_tensor = make_tensor(sentence, input_lang)
    encoded_outputs = encoder(encoder_input_tensor)

    decoder_input_tensor = make_tensor(BEGIN_TOKEN, output_lang)
    decoder_hidden = encoder.hidden
    decoded_words = []
    for i in range(MAX_LENGTH):
        output_tensor, decoder_hidden = attention_decoder(decoder_input_tensor, decoder_hidden, encoded_outputs)
        _, index = output_tensor.data[0].topk(1)
        decoded_word = output_lang.i2w[index[0]]
        decoded_words.append(decoded_word)
        if decoded_word == END_TOKEN:
            break
        decoder_input_tensor = Variable(index)

    return decoded_words

train_with_attention()
