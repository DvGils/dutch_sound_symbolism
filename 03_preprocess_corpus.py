from src.preprocessing.corpus import make
from src.preprocessing.oscar import oscar

# import os
# from filesplit.split import Split
# from src.preprocessing.sonar500 import sonar500_extractor
# from src.preprocessing.combine import corpora_combiner
# from src.preprocessing.test import pipeline_tester
# from src.preprocessing.cgnl import cgnl_extractor




"""
This script is used to extract text from the raw SoNaR-500 and CGNL data files using sonar500_exctractor() and 
cgnl_extractor(). Then, these corpora are combined with the 2018 CommonCrawl Dutch snapshot, after which the data is 
cleaned and stored as one final corpus file. Cleaning steps taken include: remove non-alphabetic characters, split into 
sentences and store these on different lines, and lowercase all words.
"""

#sonar = '/Volumes/University/TiU/Research/Resources/20150602_SoNaRCorpus_NC_1.2.1.tgz'
#sonar_out = './raw_data/corpora/extracted/sonar500/'
#snr_path = os.path.join(sonar_out, 'sonar500.txt')

#cgnl = '/Volumes/University/TiU/Research/Resources/CGNAnn2.0.3.tar.gz'
#cgn_path = './raw_data/corpora/extracted/cgnl/cgn_complete.txt'

#cc100 = '/Volumes/University/TiU/Research/Resources/cc100_dutch.tar'
#cc100_path = './raw_data/corpora/extracted/cc100_dutch.txt'

# RUN TO EXTRACT RELEVANT SONAR500 FILES FROM HUGE TARFILE
#sonar500_extractor(sonar, sonar_out)

# RUN TO EXTRACT RELEVANT CGNL FILES FROM TARFILE AND DO SOME MINOR CGNL-SPECIFIC CLEANING
#cgnl_extractor()

# RUN TO COMBINE ALL TRHEE CORPORA TOGETHER (number of lines = 261,376,546)
#corpora_combiner()

# RUN TO CREATE SMALL TEST CORPUS TO CHECK WHETHER MY PREPROCESSING PIPELINE WORKS WELL
# pipeline_tester()


# SPLITTING MASTER FILE UP INTO CHUNKS TO MAKE IT WORK TOGETHER WITH THE MAKE FUNCTION AND MULTIPROCESSING
# split = Split(
#   inputfile='./raw_data/corpora/extracted/complete_combined/corpora_complete_combined.txt',
#   outputdir='./raw_data/corpora/extracted/complete_combined/corpus_chunks/'
# ).bylinecount(linecount=270000)

# split = Split(
#     inputfile='./raw_data/corpora/extracted/nlcow.txt',
#     outputdir='./raw_data/corpora/extracted/nlcow_chunks/'
# ).bylinecount(linecount=300000)


# PREPROCESSING THE EXTRACTED SONAR500, CGN, and CC100-DUTCH / NLCoW
# './raw_data/corpora/extracted/complete_combined/corpus_chunks/'
make(corpus_dir='./raw_data/corpora/extracted/nlcow_chunks/',
     out_path='processed_data/corpus/nlcow_5+.txt',
     filter_list=[],
     min_len=5,
     threads=32)


# oscar('./processed_data/corpus/oscar_5+.txt', filter_list=[], min_len=5, threads=32)

