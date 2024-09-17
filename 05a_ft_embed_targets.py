import os
import pickle as pkl
from scipy.spatial.distance import cosine
import pandas as pd
from src.dsm import ft, utils


"""
This script is used to generate BERT and fastText embeddings for the company names, first names, and nonwords 
that we gathered data for with our survey. Furthermore, the second half of this script combines the embeddings
for the three different types of names/words together into a unified dataframe, one for fastText and one for BERT. 
"""

LEXICAL = False
m = 2
M = 5

out_dir = './processed_data/embeddings/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# LOAD DATA
# load the names/nonwords
company_df = pd.read_csv('./processed_data/words_and_names/survey_lists/company_names_final.csv')
names_df = pd.read_csv('./processed_data/words_and_names/survey_lists/dutch_names_final.csv')
pseudoword_df = pd.read_csv('./processed_data/words_and_names/survey_lists/nonwords_final.csv')

# lowercase the words and combine into a single iterable
company_names = [word[0].lower() for word in company_df.values.tolist()]
first_names = [word[0].lower() for word in names_df.values.tolist()]
pseudowords = [word[0].lower() for word in pseudoword_df.values.tolist()]
stimuli = company_names + first_names + pseudowords

# fastText
# generate embeddings using the embed() method from the src.dsm.ft module
# returns a dict of dicts where outer keys are target strings, inner keys are n-grams from the target string and
# values are the embeddings resulting from the exclusion of the ngram. the inner key 'none' indicates the embedding
# resulting from the combination of all required lexical and sub-lexical information
ft_model = 'processed_data/dsm/nlcow_5+_d300_w5_m2_M5_rho0.6109.bin'
# 'processed_data/dsm/corpus_final_d300_w5_m2_M5_rho0.5974.bin'


# attribute embeddings (anchoring the semantic differential are always derived using all available information
# target embeddings are subject to explorations about the role of sub-lexical information
stimuli_ft_embs = ft.embed(stimuli, ft_model, min_ngram=m, max_ngram=M, lexical=LEXICAL, loo=False)
with open(os.path.join(out_dir, 'ft_nlcow_5+_m{}_M{}_lex{}_embeddings_stimuli.pkl'.format(m, M, LEXICAL)), 'wb') as f:
    pkl.dump(stimuli_ft_embs, f)

# Make DFs
# create a dictionary with the embedding data that can be turned into a dataframe
# loop through company_names, first_names, and pseudowords; fetch embeddings from stimuli_bert_embs and stimuli_ft_embs;
# combine
survey_ft_embs = {
    'company': {w: stimuli_ft_embs[w] for w in company_names},
    'names': {w: stimuli_ft_embs[w] for w in first_names},
    'pseudoword': {w: stimuli_ft_embs[w] for w in pseudowords}
}

# use df_maker to combine and save the dictionaries as a dataframe
survey_ft_df = utils.df_maker(
    survey_ft_embs,
    'survey_data',
    'ft_nlcow_5+_m{}_M{}_lex{}'.format(m, M, LEXICAL),
    './processed_data/analyses/dataframes/'
)
