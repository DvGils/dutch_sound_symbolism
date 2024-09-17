import os
import pickle as pkl
import pandas as pd
from src.dsm import bert, utils
from src.analyses.correlation_analysis import get_attribute_words


"""
This script is used to generate BERT and fastText embeddings for the company names, first names, and nonwords 
that we gathered data for with our survey. Furthermore, the second half of this script combines the embeddings
for the three different types of names/words together into a unified dataframe, one for fastText and one for BERT. 
"""

out_dir = './processed_data/embeddings/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# LOAD DATA
# load the names/nonwords
company_df = pd.read_csv('./processed_data/words_and_names/survey_lists/company_names_final.csv')
names_df = pd.read_csv('./processed_data/words_and_names/survey_lists/dutch_names_final.csv')
pseudoword_df = pd.read_csv('./processed_data/words_and_names/survey_lists/nonwords_final.csv')

attributes = get_attribute_words()

# lowercase the words and combine into a single iterable
first_names = [word[0].lower() for word in names_df.values.tolist()]
company_names = [word[0].lower() for word in company_df.values.tolist()]
pseudowords = [word[0].lower() for word in pseudoword_df.values.tolist()]
stimuli = company_names + first_names + pseudowords

attribute_words = {
    w for attribute in attributes.keys() for pole in attributes[attribute].keys() for w in attributes[attribute][pole]
}

# BERT
# generate embeddings using the embed() method of the src.dsm.bert module
# returns a dict of dicts, where outer keys are strings, inner keys are indices indicating layers and values
# are embeddings derived from the target model
bert_model = 'pdelobelle/robbert-v2-dutch-base'
stimuli_bert_embs = bert.embed(stimuli, bert_model, layers=12)
attribute_bert_embs = bert.embed(attribute_words, bert_model, layers=12)
with open(os.path.join(out_dir, 'bert_embeddings_attributes.pkl'), 'wb') as f:
    pkl.dump(attribute_bert_embs, f)
with open(os.path.join(out_dir, 'bert_embeddings_stimuli.pkl'), 'wb') as f:
    pkl.dump(stimuli_bert_embs, f)

# Make DFs
# create a dictionary with the embedding data that can be turned into a dataframe
# loop through company_names, first_names, and pseudowords; fetch embeddings from stimuli_bert_embs and stimuli_ft_embs;
# combine
survey_bert_embs = {
    'company': {w: stimuli_bert_embs[w] for w in company_names},
    'names': {w: stimuli_bert_embs[w] for w in first_names},
    'pseudoword': {w: stimuli_bert_embs[w] for w in pseudowords}
}

# use df_maker to combine and save the dictionaries as a dataframe
survey_bert_df = utils.df_maker(survey_bert_embs, 'survey_data', 'bert', './processed_data/analyses/dataframes/')
