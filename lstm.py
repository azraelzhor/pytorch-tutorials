import torch
from torch.autograd import Variable

torch.manual_seed(2018)
lstm = torch.nn.LSTM(3, 3) # input dim 3, output dim 3
inputs = [Variable(torch.randn((1, 3))) for _ in range(5)]

hidden = (Variable(torch.randn(1, 1, 3)),  # h0 num_layer x batch x hidden_size
            Variable(torch.randn(1, 1, 3)))

# for i in inputs:
#     output, hidden = lstm(i.view(1, 1, -1), hidden) # output 1 x 1 x 3
#     print(output)
#     print(hidden[0])
#     print(output.size())

# inputs = torch.cat(inputs).view(-1, 1, 3)
# output, hidden = lstm(inputs, hidden) # output 5 x 1 x 3
# print(output)
# print("------")
# print(hidden[0])

training_data = [ ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
                    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
                ]

word_to_ix = {} # dictionary
tag_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
print(word_to_ix)
print(tag_to_ix)

# create model
class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tag_set):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = torch.nn.Linear(hidden_dim, tag_set)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_out = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.log_softmax(tag_out)
        return tag_scores

# train model
EMBEDDING_DIM = 10
HIDDEN_DIM = 10

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def prepare_sequence(sentence, word_to_ix):
    idxs = torch.LongTensor([word_to_ix[w] for w in sentence])
    seq = Variable(idxs.view(-1))
    return seq

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:

        model.zero_grad()
        model.hidden = model.init_hidden()

        input_var = prepare_sequence(sentence, word_to_ix)
        target_var = prepare_sequence(tags, tag_to_ix)

        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        loss.backward()
        optimizer.step()

    print('Epoch %d, loss = %f' % (epoch + 1, loss.data[0]))
