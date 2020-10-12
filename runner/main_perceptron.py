import sys
sys.path.insert(1, '../taggers/')
sys.path.insert(2, '../parser/')

import baseline_tagger
import baseline_parser
import perceptron_tagger as tagger
import perceptron_parser as parser

print("Running perceptron baseline system...")

tagger1 = tagger.train_perceptron(baseline_tagger.train_data, n_epochs=1)
parser1 = parser.train_perceptron(baseline_parser.train_data, n_epochs=1)

# If using other data, make sure to use projectivize.py
input_parser = parser.dev_data
input_tagger = tagger.dev_data

with open("output_perceptron.conllu", "w") as f:
    for elem in input_tagger:
        sen = []
        for word in elem:
            sen.append(word[0])
        pred_tags = tagger1.predict(sen)
        pred_pars = parser1.predict(sen, pred_tags)
        for index, (word, tag, par) in enumerate(zip(sen, pred_tags, pred_pars)):
            # Output this to CoNLL-U format
            f.write(str(index+1) + "    " + word + "    " + "_" + \
            "   " + tag + " " + "_" + " " + "_" + " " + str(par) + " " + "_" + \
            "   " + "_" + " " + "_" + "\n")

tag_acc = baseline_tagger.accuracy(tagger1, input_tagger)
pars_acc = baseline_parser.uas(parser1, input_parser)
print("Tagger accuracy: " + str(tag_acc))
print("UAS: " + str(pars_acc))
