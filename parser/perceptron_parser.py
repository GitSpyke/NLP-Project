from baseline_parser import *
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

class PerceptronParser(Parser):

    def __init__(self, vocab_words, vocab_tags):
        self.w2i = vocab_words
        self.t2i = vocab_tags
        self.model = Linear(len(Parser.MOVES))

    def featurize(self, words, tags, config):
        stack = config[1]
        n = config[0]
        # The features
        next_word = PAD
        top_stack_w = PAD
        second_stack_w = PAD

        next_tag = PAD
        top_stack_t = PAD
        second_stack_t = PAD

        if (len(words) > n):
            next_word = words[n]
            next_tag = tags[n]

        if(len(stack) > 0):
            top_stack_w = words[stack[-1]]
            top_stack_t = tags[stack[-1]]

        if(len(stack) > 1):
            second_stack_w = words[stack[-2]]
            second_stack_t = tags[stack[-2]]

        return [(0, next_word), (1, top_stack_w), (2, second_stack_w), (3, next_tag), (4, top_stack_t), (5, second_stack_t)]

    def predict(self, words, tags):
        config = self.initial_config(len(words))
        while(not self.is_final_config(config)):
            pred_dict = self.model.forward(self.featurize(words, tags, config))
            best_move = max(self.valid_moves(config), key=lambda x: pred_dict[x])
            config = self.next_config(config, best_move)

        return config[2]

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


######################### Training the parser ##########################
def train_perceptron(train_data, n_epochs=1):
    vocab_words, vocab_tags = make_vocabs(train_data)
    parser = PerceptronParser(vocab_words, vocab_tags)
    trainer = PerceptronTrainer(parser.model)
    for features, gold_move in samples(train_data, parser, n_epochs):
        trainer.update(features, gold_move)
    trainer.finalize()
    return parser

#parser1 = train_perceptron(train_data, n_epochs=1)
#print('{:.4f}'.format(uas(parser1, dev_data)))
