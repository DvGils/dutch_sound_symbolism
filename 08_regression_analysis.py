from src.analyses.regression_analysis import grid_search, open_processed_wordscores_rds, regression_analysis
import os
import pickle as pkl
import pandas as pd

"""
This script is used to generate the data to be used for the regression analysis, which is performed 
with script 09. 

In this script the data generated under script 05 is used. First, grid search is performed to find 
the optimal hyperparameter settings for the regression analysis. Then, the regression analysis is 
performed and dataframes with regression scores are automatically saved.
"""

data_dir = './processed_data/analyses/dataframes'

# SET IMPORTANT PARAMETERS
RND_ITERS = 2
FEATURE = 'embedding'
TARGET = 'mean_rating'
GROUP = 'word_type'
ITEM = 'name'
MODEL = 'ft'                                                                    # options: 'ft', 'bert'
LAYER = ['none']                                                                # options: ['none'] or list(range(12))
LEXICAL = True                                                                  # options: True, False
m = 0                                                                           # options: 0, 2
M = 0                                                                           # options: 0, 2, 5
LAYERS = ['none']                                                               # options: ['None'], range(12)
ASSOCIATIONS = ['gender', 'polarity', 'trustworthiness', 'smartness']
OUT_DIR = 'output/predictions/regression'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if MODEL == 'ft':
    target_df_path = os.path.join(data_dir, 'survey_data_ft_m{}_M{}_lex{}_df.pkl'.format(m, M, LEXICAL))
else:
    target_df_path = os.path.join(data_dir, 'survey_data_bert_df.pkl')

with open(target_df_path, 'rb') as f:
    target_df = pkl.load(f)


# load the survey rating scores
word_scores = open_processed_wordscores_rds()

# GRID SEARCH
# perform grid search to find the optimal hyperparameter settings per association

# FASTTEXT
# merge the survey ratings with the fasttext embedding data
df = pd.merge(target_df, word_scores, on=['name', 'word_type'])
# columns:
# - name: the target string
# - word_type: whether the name is a person name ('name'), made-up company name ('company') or pseudoword ('pseudoword')
# - embedder: which model was used to embed strings
# - layer: which layer representation was used to embed strings ('none' for fastText models)
# - embedding: a vector, the embedded name
# - mean_rating: a float, the normalized score of the name on a semantic differential
# - association: the target semantic differential (valence, gender, trustworthiness, smartness)


# perform grid search for all associations
emb_type = pd.unique(df['embedder'])[0]

for association in ASSOCIATIONS:
    print('#####association: {}; emb_model: {};'.format(association, emb_type))
    df_subset = df.loc[df['association'] == association].reset_index(drop=True)

    search_df = grid_search(df=df_subset,
                            x_col=FEATURE,
                            y_col=TARGET,
                            group_col=GROUP,
                            units=[25, 50, 150],
                            dropout=[0, 0.25, 0.5],
                            activations=['sigmoid', 'relu', 'tanh'],
                            n_layers=[1],
                            learning_rates=[0.001, 0.0001])

    # save the output as a .csv
    search_df.to_csv(
        './processed_data/analyses/grid_search/{}_{}_grid-search_first-names-lexical_True.csv'.format(
            association, emb_type
        ), index=False, index_label=False
    )

# REGRESSION ANALYSIS
# load the optimal hyperparameter settings
HYPERPARAMETERS = pd.read_csv('./results/grid_search/best_hyperparameters_first-names-lexical_True.csv')

# FASTTEXT
# subset the hyperparameters for the fasttext models
hyperparameters_ft = HYPERPARAMETERS[HYPERPARAMETERS['emb_type'] == 'ft']

# perform regression analysis with the optimal hyperparameter settings
regression_analysis(merged_ft_df, hyperparameters_ft, FEATURE, TARGET, ITEM, GROUP, ft=True)


# BERT
# subset the hyperparameters for the bert layers
hyperparameters_bert_unprocessed = HYPERPARAMETERS[HYPERPARAMETERS['emb_type'] == 'bert']

# create a dictionary with the name of each association and its corresponding best performing layer
bert_layers = {'feminine' : 1, 'good' : 10, 'smart' : 2, 'trustworthy' : 0}

# initialize an empty dataframe to store hyperparameter information in
hyperparameters_bert = pd.DataFrame(columns = hyperparameters_bert_unprocessed.columns)

# for each association, take the corresponding best-performing hyperparameter settings and save
# it to the dataframe initialized above
for key in bert_layers.keys():
    hyperparameters_bert = pd.concat([hyperparameters_bert, hyperparameters_bert_unprocessed[(hyperparameters_bert_unprocessed['association'] == key) & 
                                                                                             (hyperparameters_bert_unprocessed['emb_model'] == bert_layers[key])]])
# perform regression analysis with the optimal hyperparameter settings
regression_analysis(merged_bert_df, hyperparameters_bert, FEATURE, TARGET, ITEM, GROUP, ft=False)