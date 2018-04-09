import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(2018)

# word_to_ix = {"hello": 0, "soccer": 1, "hi": 2}
# embeds = torch.nn.Embedding(3, 5)
# lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
# hello_embed = embeds(Variable(lookup_tensor))
# print(hello_embed)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
# print(trigrams[:6])
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = torch.nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = torch.nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs

# losses = []
# criterion = torch.nn.NLLLoss()
# ngram = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = torch.optim.SGD(ngram.parameters(), lr=0.001)

# for epoch in range(100):
#     total_loss = torch.Tensor([0])
#     for context, target in trigrams:
#         context_idxs = [word_to_ix[w] for w in context]
#         context_tensor = Variable(torch.LongTensor(context_idxs))

#         optimizer.zero_grad()
#         output = ngram(context_tensor)
#         target_tensor = Variable(torch.LongTensor([word_to_ix[target]]))

#         loss = criterion(output, target_tensor)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.data
#     losses.append(total_loss)

# print(losses)


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOW(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size, context_size):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(2 * CONTEXT_SIZE * embedding_dim, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = self.linear(embeds)
        out = self.softmax(out)
        return out

# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

make_context_vector(data[0][0], word_to_ix)  # example

cbow = CBOW(EMBEDDING_DIM, vocab_size, CONTEXT_SIZE)

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=0.001)

all_losses = []

for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in data:
        context_idxs = [word_to_ix[w] for w in context]
        context_tensor = Variable(torch.LongTensor(context_idxs))
        optimizer.zero_grad()
        
        output_tensor = cbow(context_tensor)
        target_tensor = Variable(torch.LongTensor([word_to_ix[target]]))

        loss = criterion(output_tensor, target_tensor)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.data

    all_losses.append(total_loss)

print(all_losses)
