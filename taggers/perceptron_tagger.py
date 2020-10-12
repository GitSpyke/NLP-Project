from baseline_tagger import *
from collections import defaultdict

class Linear(object):

    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.weight = {c: defaultdict(float) for c in range(output_dim)}
        self.bias = {c: 0.0 for c in range(output_dim)}

    def forward(self, features):
        output_vector = dict()

        for class_type in self.weight.keys():
            result = 0.0
            for feature in features:
                if(feature in self.weight[class_type].keys()):
                    result += self.weight[class_type][feature]
            output_vector[class_type] = result + self.bias[class_type]

        return output_vector

class PerceptronTagger(Tagger):

    def __init__(self, vocab_words, vocab_tags):
        self.lin = Linear(len(vocab_tags))
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, words, i, pred_tags):
        tot = [0, 0, 0, 0]
        tot[0] = (0, words[i])
        if(i > 0):
            tot[1] = (1, words[i-1])
            tot[3] = (3, pred_tags[i-1])
        else:
            tot[1] = (1, PAD)
            tot[3] = (3, PAD)

        if(i < len(words)-1):
            tot[2] = (2, words[i+1])
        else:
            tot[2] = (2, PAD)
        return tot

    def predict(self, words):
        pred = []
        for i in range(len(words)):
            feat = self.featurize(words, i, pred)
            res = self.lin.forward(feat)
            index = max(res, key=res.get)
            for key, value in self.vocab_tags.items():
                if index == value:
                    pred.append(key)
        return pred


######################### Perceptron trainer ##########################
class PerceptronTrainer(object):

    def __init__(self, model):
        self.model = model
        self.acc = Linear(model.output_dim)
        self.counter= 1

    def update(self, features, gold):
        total_weights = self.model.forward(features)
        p = max(total_weights, key=total_weights.get)
        if (not (p == gold)):
            # Assume features are represented by ints
            for word in features:
                self.model.weight[p][word] -= 1
                self.model.weight[gold][word] += 1
                self.acc.weight[p][word] -= self.counter
                self.acc.weight[gold][word] += self.counter
            self.model.bias[p] -= 1
            self.model.bias[gold] += 1
            self.acc.bias[p] -= self.counter
            self.acc.bias[gold] += self.counter
        self.counter += 1

    def finalize(self):
        for i in range(0, len(self.model.weight)):
            for key in self.model.weight[i].keys():
                self.model.weight[i][key] -= self.acc.weight[i][key] / self.counter
            self.model.bias[i] -= self.acc.bias[i] / self.counter


################ Training perceptron ########################
import random

def train_perceptron(train_data, n_epochs=1):
    vocab_words, vocab_tags = make_vocabs(train_data)
    train_data = train_data[:]
    tagger = PerceptronTagger(vocab_words, vocab_tags)
    trainer = PerceptronTrainer(tagger.lin)
    for _ in range(n_epochs):
        random.shuffle(train_data)

        for sen in train_data:
            words, tags = zip(*sen)
            pred = []
            for j in range(len(words)):
                feat = tagger.featurize(words, j, pred)
                trainer.update(feat, vocab_tags[tags[j]])
                pred.append(tags[j])
            words = []
            tags = []

    trainer.finalize()
    return tagger

#tagger4 = train_perceptron(train_data, n_epochs=1)
#print('{:.4f}'.format(accuracy(tagger4, dev_data)))
