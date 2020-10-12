from baseline_parser import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim):
        super().__init__() # To init nn.module
        self.output_dim = output_dim
        self.emb_specs = nn.ModuleList()
        self.embedding_size = 0
        for i in range(len(embedding_specs)):
            self.embedding_size += embedding_specs[i][1]
            self.emb_specs.append(nn.Embedding(embedding_specs[i][0], embedding_specs[i][1]))

        for spec in self.emb_specs:
            nn.init.normal_(spec.weight, std=(10)**(-2))
            # Uncomment code below to get weight tying
            # spec.weight = self.emb_specs[0].weight

        #n_words = embedding_specs[-1][0]
        #tag_dim = embedding_specs[-1][1]
        #self.tag_embedding = nn.Embedding(n_words, tag_dim)

        #nn.init.normal_(self.tag_embedding.weight,std=(10)**(-2))
        self.linear = nn.Linear(self.embedding_size, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        #if (len(features[0]) == 1):
        #    splitted = torch.split(features, 1)
        #else:
        splitted = torch.split(features, 1, dim=1)

        linear_list = list()
        for i,spec in enumerate(self.emb_specs):
            emb = spec(splitted[i])
            linear_list.append(emb)

        #linear_list.append(self.tag_embedding(splitted[-1]))
        linear = self.linear(torch.cat(tuple(linear_list), 2))
        relu = self.relu(linear)
        output = self.output(relu)
        return output.view(-1, self.output_dim)

class NeuralParser(Parser):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=50, hidden_dim=200):
        self.w2i = vocab_words
        self.t2i = vocab_tags
        embedding_specs = [(len(vocab_words), word_dim)] * 3 + [(len(vocab_tags), tag_dim)] * 3
        self.model = Network(embedding_specs, hidden_dim, len(Parser.MOVES))
        #for e in self.model.emb_specs:
        #    nn.init.normal_(e.weight, mean=0, std=1e-2)

    def featurize(self, words, tags, config):
        stack = config[1]
        n = config[0]
        # The features
        next_word = 0
        top_stack_w = 0
        second_stack_w = 0

        next_tag = 0
        top_stack_t = 0
        second_stack_t = 0

        if (len(words) > n):
            next_word = words[n]
            next_tag = tags[n]

        if(len(stack) > 0):
            top_stack_w = words[stack[-1]]
            top_stack_t = tags[stack[-1]]

        if(len(stack) > 1):
            second_stack_w = words[stack[-2]]
            second_stack_t = tags[stack[-2]]

        return_tensor = torch.Tensor([next_word, top_stack_w, second_stack_w, next_tag, top_stack_t, second_stack_t]).long()

        return return_tensor

    def predict(self, words, tags):
        word_num = []
        tag_num = []
        for word,tag in zip(words, tags):
            if (word in self.w2i.keys()):
                word_num.append(self.w2i[word])
            else:
                word_num.append(1)

            if (tag in self.t2i.keys()):
                tag_num.append(self.t2i[tag])
            else:
                tag_num.append(1)

        config = self.initial_config(len(words))
        while(not self.is_final_config(config)):
            feature = self.featurize(word_num,tag_num, config)
            squeezed = torch.unsqueeze(feature, 0)
            best_move = self.model.forward(squeezed).argmax().item()
            config = self.next_config(config, best_move)
        return config[2]


def encode(gold_data, vocab_words, vocab_tags):
    encoded_data = []
    for sentence in gold_data:
        encoded_sentence = []
        for word, tag, head in sentence:
            word = word if word in vocab_words else UNK
            encoded_word = vocab_words[word]
            encoded_tag = vocab_tags[tag]
            encoded_sentence.append((encoded_word, encoded_tag, head))
        encoded_data.append(encoded_sentence)
    return encoded_data


def train_neural(train_data, n_epochs=1, batch_size=300):
    vocab_words, vocab_tags = make_vocabs(train_data)
    train_data = encode(train_data, vocab_words, vocab_tags)
    parser = NeuralParser(vocab_words, vocab_tags)
    optimizer = optim.Adam(parser.model.parameters())
    bx, by = [], []
    for features, gold_move in samples(train_data, parser, n_epochs):
        bx.append(features)
        by.append(gold_move)
        if len(bx) >= batch_size:
            bx = torch.stack(bx)
            by = torch.LongTensor(by)
            optimizer.zero_grad()
            output = parser.model.forward(bx)
            loss = F.cross_entropy(output, by)
            loss.backward()
            optimizer.step()
            bx, by = [], []
    return parser

#parser2 = train_neural(train_data[:6000], n_epochs=1)
#print('{:.4f}'.format(uas(parser2, dev_data)))
