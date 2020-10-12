# TDDE09-project

## Files
The project is divided into three folders, where tagger/ contains the perceptron and the neural tagger, parser/ contains the perceptron and neural parser and runner/ contains the files to run the entire program. 

## Running the code
Make sure you have activated a python environment with torch installed. Put the conllu files you want to run in the runner/ folder. If the treebanks contains non-projective trees, first projectivize the train data.
```bash
python3 projectivize.py < train_data_file.conllu > new_name_of_file.conllu
```
To run the program with the intended files, the file names in the beginning of both baseline_tagger.py and baseline_parser.py need to be changed so they open the correct files. Then, the program is run with:
Perceptron
```bash
python3 main_perceptron.py
```
Neural
```bash
python3 main_neural.py
```
The resulting tagger accuracy and UAS is printed to the console. The output is written to output_perceptron.conllu and output_neural.conllu respectively.

