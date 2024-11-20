from datasets import load_dataset
from util import is_not_german
import random
from torch import nn
import torch

ds = load_dataset("wmt/wmt19", "fr-de")

underCons = ds["train"][0:100000]["translation"]

numSentences = len(underCons)
print(numSentences)
filterOutStrangeFilter = filter(lambda x: not is_not_german(
    x["de"]), underCons)
filterOutStrange = list(filterOutStrangeFilter)
print("num filtered out")
print(filterOutStrange[0]["de"])
numSentencesAfterFiler = len(filterOutStrange)
print("numSentencesAfterFilter: " + str(numSentencesAfterFiler))
deTrain = list(map(lambda x: x["de"], filterOutStrange))
deTrainMerged = "".join(deTrain)
vocab = sorted(set(deTrainMerged))
print(vocab)
print(len(vocab))

random_int = random.randint(0, len(deTrain))
print(deTrain[random_int])

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] + 1 for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i-1] for i in l])


print(encode("hii there"))
print(decode(encode("hii there")))

half_embedding_dim = 8
embedding_dim = 2 * half_embedding_dim
vocab_size = len(vocab) + 1


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_pos=1024):
        pos_matrix = torch.arange(max_pos).repeat(half_embedding_dim)

        base = 10000

        index_exponent_matrix = (
            2.0 * torch.arange(half_embedding_dim).view(-1, 1).repeat(1, max_pos)) / embedding_dim

        index_matrix = torch.pow(base, index_exponent_matrix)

        pos_encoding = torch.zeros(max_pos, embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(pos_matrix * index_matrix)
        pos_encoding[:, 1::2] = torch.cos(pos_matrix * index_matrix)

    def forward(self, input):


class MyDecoderModule(nn.Module):

    def __init__(self):
        super(MyDecoderModule, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, encoded, decoded):
        encoded_embedded = self.embedding(encoded)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
