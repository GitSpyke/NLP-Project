from baseline_tagger import *
import torch
import torch.nn

class Network(torch.nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim):
        super().__init__()
        # Initialize word embeddings
        self.embeddings = torch.nn.ModuleList()
        hidden_in_dim = 0
        for ind, spec in enumerate(embedding_specs):
            emb = torch.nn.Embedding(spec[0], spec[1])
            torch.nn.init.normal_(emb.weight, mean=0, std=10**(-2))
            self.embeddings.append(emb)
            hidden_in_dim += spec[1]

        self.hidden = torch.nn.Linear(hidden_in_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        if features.size() == torch.Size([4]):
            feats = features
            embeds = torch.zeros(1, 0)
            sample_size = 1
        else:
            feats = torch.split(features, 1, dim=1)
            hidden_dim = self.hidden.weight.size()[0]
            embeds = torch.zeros(hidden_dim, 0)
            sample_size = hidden_dim

        index = 0
        for emb in self.embeddings:
            embedding = emb(feats[index])

            if sample_size == 1:
                embeds = torch.cat((embeds, embedding.unsqueeze(0)), 1)
            else:
                embeds = torch.cat((embeds, embedding.view(sample_size, -1)), 1)
            index += 1

        hid = self.hidden(embeds)
        rel = self.relu(hid)
        output = self.out(rel)
        return output


class NeuralTagger(Tagger):

    def __init__(self, vocab_words, vocab_tags):
        self.w2i = vocab_words
        self.i2t = {}
        for key, value in vocab_tags.items():
            self.i2t[value] = key
        embedding_specs = [(len(vocab_words), 50)] * 3 + [(len(vocab_tags), 10)]
        self.model = Network(embedding_specs, 100, len(vocab_tags))

    def featurize(self, words, i, pred_tags):
        tot = [0, 0, 0, 0]
        tot[0] = words[i]
        if(i > 0):
            tot[1] = words[i-1]
            tot[3] = pred_tags[i-1]
        else:
            tot[1] = 0 # pad
            tot[3] = 0

        if(i < len(words)-1):
            tot[2] = words[i+1]
        else:
            tot[2] = 0
        return torch.Tensor(tot).long()

    def predict(self, words):
        word_indices = []

        for w in words:
            if w not in self.w2i:
                word_indices.append(1)
            else:
                word_indices.append(self.w2i[w])
        pred_tags = []
        pred_tags_index = []
        for i in range(len(words)):
            feat = self.featurize(word_indices, i, pred_tags_index)
            res = self.model.forward(feat)
            max_ind = torch.argmax(res).item()
            pred_tags_index.append(max_ind)
            pred_tags.append(self.i2t[max_ind])

        return pred_tags


#################### Training neural ################
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim

def batchify(gold_data, batch_size, tagger, vocab_words, vocab_tags):
    feats = []
    tot_tags = []
    for sen in gold_data:
        sen_words_indices = []
        sen_tags_indices = []
        for word, tag in sen:
            sen_words_indices.append(vocab_words[word])
            tot_tags.append(vocab_tags[tag])

        pred_tags = []
        for ind, word in enumerate(sen):
            feat = tagger.featurize(sen_words_indices, ind, pred_tags)
            feats.append(feat)
            sen_tags_indices.append(vocab_tags[word[1]])
            pred_tags.append(vocab_tags[word[1]])

    random_indices = torch.randperm(len(feats))
    stack = torch.stack(feats).long()
    answer_tags = torch.Tensor(tot_tags).long()

    for i in range(0, len(stack) - batch_size + 1, batch_size):
        indices = random_indices[i:i+batch_size]
        yield stack[indices], answer_tags[indices]

def train_neural(train_data, n_epochs=1, batch_size=100):
    vocab_words, vocab_tags = make_vocabs(train_data)
    train_data = train_data[:]
    tagger = NeuralTagger(vocab_words, vocab_tags)
    optimizer = optim.Adam(tagger.model.parameters())
    for _ in range(n_epochs):
        for bx, by in batchify(train_data, batch_size, tagger, vocab_words, vocab_tags):
            optimizer.zero_grad()
            output = tagger.model.forward(bx)
            loss = F.cross_entropy(output, by)
            loss.backward()
            optimizer.step()
    return tagger

#tagger5 = train_neural(train_data[:6000], n_epochs=1)
#print('{:.4f}'.format(accuracy(tagger5, dev_data)))
