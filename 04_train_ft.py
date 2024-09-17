import pandas as pd
from src.dsm.ft import grid

"""
This script can be used to train fasttext models using the cleaned corpus file. We provide code to train a purely 
lexical model (analogue to word2vec) and a model combining lexical and sub-lexical information (2-, 3-, 4-, and 
5-character ngrams). Models are saved as .bin files in the form of 
'[corpus name]_d[dimensionality]_w[window size]_m[minimum n-gram size]_M[maximum n-gram size].bin'. Lexical models have
a min and max ngram size of 0, indicating that no sub-lexical information is considered.
Models are primarily benchmarked against the Dutch ws-353 dataset from https://aclanthology.org/L18-1618.pdf .
Additional supported benchmarks include the Dutch SimLex-999 dataset from the same source, and the relatedness scores
from the SICK-NL dataset, cf: https://github.com/gijswijnholds/sick_nl/tree/master .
"""


ws353 = pd.read_csv('./raw_data/benchmarks/nl-ws353.dataset', header=0, sep=';', index_col=0)
ws353 = ws353.rename(columns={'word1': 'w1', 'word2': 'w2', 'score': 'sim'})

simlex = pd.read_csv('./raw_data/benchmarks/nl-simlex.dataset', header=0, sep=';', index_col=0)
simlex = simlex.rename(columns={'word1': 'w1', 'word2': 'w2', 'score': 'sim'})

sick = pd.read_csv(
     './raw_data/benchmarks/SICK_NL.txt', header=0, sep='\t', index_col=0
)[['sentence_A', 'sentence_B', 'relatedness_score']]
sick = sick.rename(columns={'sentence_A': 'w1', 'sentence_B': 'w2', 'relatedness_score': 'sim'})


grid(corpus='./processed_data/corpus/sonar_cc_cgnl_5+.txt',
     eval_words=ws353,
     out_path='./processed_data/dsm/',
     dimensionalities=[300],
     window_sizes=[5],
     min_values=[2],
     max_values=[5],
     threads=64)
