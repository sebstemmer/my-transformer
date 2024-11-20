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
max_pos = 1024


class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()

        pos_matrix = torch.arange(max_pos).view(
            max_pos, -1).repeat(1, half_embedding_dim)

        print("pos_matrix")
        print(pos_matrix.size())

        base = 10000

        index_exponent_matrix = (
            2.0 * torch.arange(half_embedding_dim).repeat(max_pos, 1)) / embedding_dim

        index_matrix = torch.pow(base, index_exponent_matrix)

        weights = torch.zeros(max_pos, embedding_dim)

        weights[:, 0::2] = torch.sin(pos_matrix * index_matrix)
        weights[:, 1::2] = torch.cos(pos_matrix * index_matrix)

        self.pos_embedding = nn.Embedding.from_pretrained(weights)

    def forward(self, input):
        if len(input) > max_pos:
            raise Exception("size > max_pos")

        positions = torch.arange(len(input))

        return self.pos_embedding(positions)


class MyDecoderModule(nn.Module):

    def __init__(self):
        super(MyDecoderModule, self).__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.pos_embedding = PositionalEmbedding()

        # self.linear1 = torch.nn.Linear(100, 200)
        # self.activation = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(200, 10)
        # self.softmax = torch.nn.Softmax()

    def forward(self, encoded):  # , decoded):
        encoded_token_embedded = self.token_embedding(encoded)
        encoded_pos_embedded = self.pos_embedding(encoded)

        encoded_embedded = encoded_token_embedded + encoded_pos_embedded

        return encoded_embedded

        # x = self.linear1(x)
        # x = self.activation(x)
        # x = self.linear2(x)
        # x = self.softmax(x)


encoded = torch.tensor(encode("hii there"))
print(encoded)

myDecoderModule = MyDecoderModule()

afterForward = myDecoderModule(encoded)
print(afterForward)
print(afterForward.size())
