# This script is used to perform statistical analyses using the cosine score 
# data generated using the previous script 06

library(readr)
library(stringr)
library(dplyr)
library(randomForest)
library(ggplot2)
library(plyr)

## LOAD DATA
setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2022_2023-SoDa/dutchNames/processed_data/survey_ratings/')
survey_ratings <- readRDS('word_scores.rds')

# change the direction of the 'slecht' ratings since the cosine scores measure 
#'good'-ness, and lowercast the word column
survey_ratings$mean[survey_ratings$association == 'slecht'] <- 
  as.numeric(-survey_ratings$mean[survey_ratings$association == 'slecht'])
survey_ratings$word <- tolower(survey_ratings$word)

# create a mapping of words to be changed in the survey_ratings data to make it
# more compatible with the cosine scores data
word_map <- c('vrouwelijk' = 'gender',
              'slecht' = 'polarity',            # I understand this is not a translation, direction
              'slim' = 'smartness',             # of mean rating will be changed accordingly
              'betrouwbaar' = 'trustworthiness',
              'bedrijfsnamen' = 'company_names',
              'namen' = 'person_names',
              'nepwoorden' = 'pseudowords')
survey_ratings[, 11:13] <- mutate_all(survey_ratings[, 11:13], ~ str_replace_all(., word_map))

# rename the word_type column 
colnames(survey_ratings)[13] <- 'word_type'
colnames(survey_ratings)[12] <- 'semantic_differential'
colnames(survey_ratings)[11] <- 'target'

# load associations
setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2022_2023-SoDa/dutchNames/output/predictions/associations/')
ft_lex = read.csv('ft_m0_M0_lexTrue_associations.csv', header = T, sep = '\t')
ft_2gr = read.csv('ft_m2_M2_lexFalse_associations.csv', header = T, sep = '\t')
ft_ngr = read.csv('ft_m2_M5_lexFalse_associations.csv', header = T, sep = '\t')
ft = rbind(ft_lex, ft_2gr, ft_ngr)
rm(ft_2gr, ft_lex, ft_ngr)

ft$target = factor(ft$target)
ft$semantic_differential = factor(ft$semantic_differential)
ft$pole = factor(ft$pole)
ft$word = factor(ft$word)
ft$embedder = factor(ft$embedder)
ft$layer = factor(ft$layer)
ft$lexical = factor(ft$lexical)
ft$M = factor(ft$M)
ft$m = factor(ft$m)


ft_df = merge(
  select(survey_ratings, mean, target, semantic_differential, word_type), 
  ft, 
  by.x = c('target', 'semantic_differential'), 
  by.y = c('target', 'semantic_differential')
)


ft_aggr = ft_df %>%
  group_by(target, semantic_differential, pole, embedder, layer, lexical, m, M, mean, word_type) %>%
  summarise(avg_sim = mean(sim)) %>%
  pivot_wider(names_from = pole, values_from = avg_sim)


ft_aggr$gender_delta = ft_aggr$female - ft_aggr$male
ft_aggr$polarity_delta = ft_aggr$good - ft_aggr$bad
ft_aggr$smartness_delta = ft_aggr$smart  - ft_aggr$dumb
ft_aggr$trustworthiness_delta = ft_aggr$trustworthy - ft_aggr$untrustworthy

ft_aggr_long = ft_aggr %>%
  select(target, semantic_differential, embedder, layer, lexical, m, M, mean, word_type, gender_delta, polarity_delta, smartness_delta, trustworthiness_delta) %>%
  pivot_longer(cols = ends_with('_delta'), names_to = "semantic_delta", values_to = "delta_value", values_drop_na = TRUE)

ft_aggr_long = unite(ft_aggr_long, col='embedding_model', c('embedder', 'lexical', 'm', 'M', 'layer'), sep='-')
ft_aggr_long$target = factor(ft_aggr_long$target)
ft_aggr_long$semantic_differential = factor(ft_aggr_long$semantic_differential)
ft_aggr_long$embedding_model = factor(ft_aggr_long$embedding_model)
ft_aggr_long$word_type = factor(ft_aggr_long$word_type)
ft_aggr_long$embedding_model = mapvalues(ft_aggr_long$embedding_model, 
                                         from = c("ft-False-2-2-full", "ft-False-2-5-full", "ft-True-0-0-full"), 
                                         to = c("ft_bigrams", "ft_ngrams", "ft_lexical"))


ggplot(data = ft_aggr_long, aes(x = delta_value, y = mean, color = embedding_model, fill = embedding_model)) +
  geom_point(size = 0.5) +
  geom_smooth(method = 'lm') +
  scale_color_manual(labels = c("ft_bigrams", "ft_ngrams", "ft_lexical"), values = c('steelblue', 'firebrick', 'darkgoldenrod2')) +
  scale_fill_manual(labels = c("ft_bigrams", "ft_ngrams", "ft_lexical"), values = c('steelblue', 'firebrick', 'darkgoldenrod2')) +
  labs(x = "semantic delta", y = "normalized rating") +
  facet_grid(word_type ~ semantic_differential)


fem_bert <- read_csv(
  paste(path, 'feminine_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''), show_col_types = FALSE
  )
good_bert <- read_csv(
  paste(path, 'good_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''), show_col_types = FALSE
  )
smart_bert <- read_csv(
  paste(path, 'smart_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''), show_col_types = FALSE
  )
trust_bert <- read_csv(
  paste(path, 'trustworthy_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''), show_col_types = FALSE
  )










ggplot(data = attr_df, aes(x = mean, y = delta_all_names, color = model, fill = model)) +
  geom_point(size = 0.5) +
  geom_smooth(method = 'lm') +
  scale_color_manual(labels = c("FT lexical", "FT n-gram"), values = c('#CDCDCD', '#F5C342')) +
  scale_fill_manual(labels = c("FT lexical", "FT n-gram"), values = c('#CDCDCD', '#F5C342')) +
  labs(y = "semantic delta", x = "normalized rating") +
  scale_y_continuous(breaks=c(-0.15, -0.05, 0.05)) +
  facet_grid(word_type ~ association) +
  theme(axis.line=element_line(color='#F2F2F2'),
        axis.text=element_text(color='#F2F2F2', size = 16),
        axis.title=element_text(color='#F2F2F2', size = 20, face="bold"),
        axis.ticks=element_line(color='#F2F2F2'),
        strip.background=element_rect(fill='#F2F2F2'),
        strip.text=element_text(color='#242A34', size = 20, face="bold"),
        panel.background = element_rect(fill='transparent'),
        plot.background = element_rect(fill='transparent', color=NA),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        legend.background = element_rect(fill='transparent'), 
        legend.box.background = element_rect(fill='transparent'),
        legend.justification = c(0, 1),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.text=element_text(color='#F2F2F2', size = 20, face="bold"),
        legend.title=element_text(color='#F2F2F2', size = 20)
  )


#colors:
# blue: 36 42 52 -> #242A34
# grey: 205 205 205 -> #CDCDCD
# yellow: 245 195 66 -> #F5C342
# white: 242 242 242 -> #F2F2F2
# 


##  CORRELATION ANALYSIS
# initialize some lists to create for-loops with
ratings_list = list(fem_ratings, good_ratings, smart_ratings, trust_ratings)
associations <- list('feminine', 'good', 'smart', 'trustworthy')
word_types <- list('company', 'fnames', 'nonword')

# FastText
cosines_list_ft = list(fem_ft, good_ft, smart_ft, trust_ft)
ft_models <- list('0', '2-5') 

# initialize an empty dataframe to store results in
correlations_ft = data.frame(matrix(ncol = 5, nrow = 0))
colnames(correlations_ft) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (i in 1:length(associations)) {
  for (wordtype in word_types) {
    for (modeltype in ft_models){
      # for every association-word_type-ft_model combination, fetch the relevant 
      # survey rating data and cosine data
      association <- associations[i]
      
      temp_ratings <- subset(data.frame(ratings_list[i]), word_type == wordtype)
      
      temp_data <- data.frame(cosines_list_ft[i])
      temp_data <- unique(subset(temp_data, 
                                 word_type == wordtype & model == modeltype, 
                                 select = c('name', 'delta_all_names')))
      
      # merge the two dataframes
      cor_df <- merge(temp_ratings, temp_data, by.x = 'word', by.y = 'name')
      
      # perform correlation analysis on the survey ratings and cosine scores
      corr <- cor(cor_df$mean, 
                  cor_df$delta_all_names)
      
      # save the correlation for this association-word_type-ft_model combination
      # in the results dataframe
      correlations_ft[nrow(correlations_ft)+1, ] <- c('ft', association, wordtype, modeltype, corr)
    
    }
  }
}  

# save the correlation results as a .csv file
write.csv(correlations_ft, 
          file = paste(path_results, 'correlations_ft0_&_2-5_bootstrap=False.csv', sep = ''), 
          row.names = FALSE)


# BERT
cosines_list_bert = list(fem_bert, good_bert, smart_bert, trust_bert)
bert_models <- list('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11') 

# initialize an empty dataframe to store results in
correlations_bert = data.frame(matrix(ncol = 5, nrow = 0))
colnames(correlations_bert) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (i in 1:length(associations)) {
  for (wordtype in word_types) {
    for (modeltype in bert_models){
      # for every association-word_type-bert_model combination, fetch the relevant 
      # survey rating data and cosine data
      association <- associations[i]
      
      temp_ratings <- subset(data.frame(ratings_list[i]), word_type == wordtype)
      
      temp_data <- data.frame(cosines_list_bert[i])
      temp_data <- unique(subset(temp_data, 
                                 word_type == wordtype & model == modeltype, 
                                 select = c('name', 'delta_all_names')))
      
      # merge the two dataframes
      cor_df <- merge(temp_ratings, temp_data, by.x = 'word', by.y = 'name')
      
      # perform correlation analysis on the survey ratings and cosine scores
      corr <- cor(cor_df$mean, 
                  cor_df$delta_all_names)
      
      # save the correlation for this association-word_type-ft_model combination
      # in the results dataframe
      correlations_bert[nrow(correlations_bert)+1, ] <- c('bert', association, wordtype, modeltype, corr)
      
    }
  }
}  

# save the correlation results as a .csv file
write.csv(correlations_bert, 
          file = paste(path_results, 'correlations_bert-all-layers_bootstrap=False.csv', sep = ''), 
          row.names = FALSE)


## EXTRA PROCESSING OF DATA
# join the fasttext and bert correlation results together into one dataframe
correlations_all <- full_join(correlations_ft, correlations_bert)

# save the combined correlation results as a .csv file
write.csv(correlations_all, 
          file = paste(path_results, 'correlations_ft0&2-5_&_bert-all-layers_bootstrap=False.csv', sep = ''),
          row.names = FALSE)

# find the highest correlation score per embedding type (bert/fasttext) - 
# association - word type combination 
correlations_highest <- correlations_all %>%
  dplyr::group_by(emb_type, association, word_type) %>%
  dplyr::slice_max(correlation)

# create a unified fastText and BERT cosine score dataframe, add and rename some
# columns for better interpretability, factorize columns with text data, and 
# save it as a .rds file
similarity_delta_df <-
  bind_rows(
    fasttext = bind_rows(cosines_list_ft),
    bert = bind_rows(cosines_list_bert) %>%
      mutate(model = as.character(model)),
    .id = "model_type"
  ) %>%
  rename(similarity_delta = delta_all_names) %>%
  mutate(across(where(is.character), as_factor)) 

write_rds(similarity_delta_df, file=paste0(path_results, 'similarity_delta_df.rds'))


correlations_bert$model = as.numeric(correlations_bert$model)
correlations_bert$word_type = factor(correlations_bert$word_type)
levels(correlations_bert$word_type)[levels(correlations_bert$word_type)=="fnames"] <- "names"

correlations_ft$word_type = factor(correlations_ft$word_type)
levels(correlations_ft$word_type)[levels(correlations_ft$word_type)=="fnames"] <- "names"

ggplot(data = correlations_bert, aes(x = model, y = correlation)) +
  geom_line(aes(color = 'BERT'), linewidth = 1.5) +
  geom_hline(data = correlations_ft[correlations_ft$model == '2-5', ], 
             aes(yintercept = correlation, color = 'FT n-gram'), linewidth = 1.5, linetype = 'dashed') +
  geom_hline(data = correlations_ft[correlations_ft$model == '0', ], 
             aes(yintercept = correlation, color = 'FT lexical'), linewidth = 1.5, linetype = 'dotted') +
  facet_grid(association ~ word_type) +
  labs(y = "correlation", x = "layer") +
  scale_color_manual(name='Model',
                     breaks=c('BERT', 'FT lexical', 'FT n-gram'),
                     values=c('BERT'='aquamarine', 'FT lexical'='#CDCDCD', 'FT n-gram'='#F5C342')) +
  theme(axis.line=element_line(color='#F2F2F2'),
        axis.text=element_text(color='#F2F2F2', size = 16),
        axis.title=element_text(color='#F2F2F2', size = 20, face="bold"),
        axis.ticks=element_line(color='#F2F2F2'),
        strip.background=element_rect(fill='#F2F2F2'),
        strip.text=element_text(color='#242A34', size = 20, face="bold"),
        panel.background = element_rect(fill='transparent'),
        plot.background = element_rect(fill='transparent', color=NA),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        legend.background = element_rect(fill='transparent'), 
        legend.box.background = element_rect(fill='transparent'),
        legend.justification = c(1, 0),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.text=element_text(color='#F2F2F2', size = 20, face="bold"),
        legend.title=element_text(color='#F2F2F2', size = 20)
  )
