import os
import pickle as pkl
import pandas as pd
from src.dsm.evaluations import pairwise_similarities
from src.analyses.correlation_analysis import get_attribute_words


"""
This script is used to generate the data to be used for the correlation analysis, which is performed 
with script 07. In this script, the data generated under the previous script 05 is used, and cosine
analysis is performed. Dataframes with cosine scores are automatically saved.
"""

# COMBINATIONS TO DO:
# - m:2 & M: 2, LEXICAL: False

# COMBINATIONS DONE:
# - m:2 & M: 5, LEXICAL: True
# - m:0 & M: 0, LEXICAL: True

emb_dir = './processed_data/embeddings'

MODEL = 'bert'                                                                      # options: 'ft', 'bert'
LEXICAL = None                                                                      # options: True, False
m = None                                                                            # options: 0, 2
M = None                                                                            # options: 0, 2, 5
LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]                                 # options: ['None'], range(12)
ATTRIBUTES = ['gender', 'polarity', 'trustworthiness', 'smartness']
OUT_DIR = 'output/predictions/associations'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if MODEL == 'ft':
    target_embeddings_path = os.path.join(emb_dir, 'ft_nlcow_5+_m{}_M{}_lex{}_embeddings_stimuli.pkl'.format(m, M, LEXICAL))
    attribute_embeddings_path = os.path.join(emb_dir, 'ft_embeddings_attributes.pkl')
else:
    target_embeddings_path = os.path.join(emb_dir, 'bert_embeddings_stimuli.pkl')
    attribute_embeddings_path = os.path.join(emb_dir, 'bert_embeddings_attributes.pkl')

with open(target_embeddings_path, 'rb') as f:
    target_embeddings = pkl.load(f)
with open(attribute_embeddings_path, 'rb') as f:
    attribute_embeddings = pkl.load(f)

attribute_dict = get_attribute_words()

associations_df = []
for attribute in ATTRIBUTES:
    print('##############{}##############'.format(attribute))

    for layer in LAYERS:
        associations = pairwise_similarities(target_embeddings, attribute_embeddings, layer)

        for t in target_embeddings.keys():
            for p, words in attribute_dict[attribute].items():
                for w in words:
                    associations_df.append({
                        'target': t,
                        'semantic_differential': attribute,
                        'pole': p,
                        'word': w,
                        'embedder': MODEL,
                        'layer': layer,
                        'lexical': LEXICAL,
                        'm': m,
                        'M': M,
                        'sim': associations[t][w]
                    })

pd.DataFrame(
    associations_df
).to_csv(
    os.path.join(OUT_DIR, '{}_m{}_M{}_lex{}_associations.csv'.format(MODEL, m, M, LEXICAL)),
    sep='\t', index=False, index_label=False
)
