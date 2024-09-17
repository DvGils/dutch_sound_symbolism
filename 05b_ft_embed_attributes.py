import os
import pickle as pkl
from src.dsm import ft
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
attributes = get_attribute_words()
attribute_words = {
    w for attribute in attributes.keys() for pole in attributes[attribute].keys() for w in attributes[attribute][pole]
}

# fastText
# generate embeddings using the embed() method from the src.dsm.ft module
# returns a dict of dicts where outer keys are target strings, inner keys are n-grams from the target string and
# values are the embeddings resulting from the exclusion of the ngram. the inner key 'none' indicates the embedding
# resulting from the combination of all required lexical and sub-lexical information
ft_model = 'processed_data/dsm/nlcow_5+_d300_w5_m2_M5_rho0.6109.bin'

# attribute embeddings (anchoring the semantic differential are always derived using all available information
# target embeddings are subject to explorations about the role of sub-lexical information
attributes_ft_embs = ft.embed(attribute_words, ft_model, min_ngram=2, max_ngram=5, lexical=True, loo=False)
with open(os.path.join(out_dir, 'ft_embeddings_attributes.pkl'), 'wb') as f:
    pkl.dump(attributes_ft_embs, f)
