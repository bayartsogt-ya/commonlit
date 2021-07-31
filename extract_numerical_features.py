import pandas as pd
import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag, pos_tag_sents
import string


import readability
import spacy
# import textstat

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import xgboost as xgb
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def readability_measurements(passage: str):
    """
    This function uses the readability library for feature engineering.
    It includes textual statistics, readability scales and metric, and some pos stats
    """
    results = readability.getmeasures(passage, lang='en')
    words_count = results['sentence info']['words']
    chars_per_word = results['sentence info']['characters_per_word']
    syll_per_word = results['sentence info']['syll_per_word']
    words_per_sent = results['sentence info']['words_per_sentence']
    sentences_per_paragraph = results['sentence info']['sentences_per_paragraph']
    type_token_ratio = results['sentence info']['type_token_ratio']
    syllables = results['sentence info']['syllables']
    wordtypes = results['sentence info']['wordtypes']
    word_diversity = wordtypes/words_count
    sentences = results['sentence info']['sentences']
    paragraphs = results['sentence info']['paragraphs']
    long_words = results['sentence info']['long_words']
    complex_words = results['sentence info']['complex_words'] 
    complex_words_dc = results['sentence info']['complex_words_dc'] 
    #14
    
    kincaid = results['readability grades']['Kincaid']
    ari = results['readability grades']['ARI']
    coleman_liau = results['readability grades']['Coleman-Liau']
    flesch = results['readability grades']['FleschReadingEase']
    gunning_fog = results['readability grades']['GunningFogIndex']
    lix = results['readability grades']['LIX']
    smog = results['readability grades']['SMOGIndex']
    rix = results['readability grades']['RIX']
    dale_chall = results['readability grades']['DaleChallIndex']
    #9
    tobeverb = results['word usage']['tobeverb']
    auxverb = results['word usage']['auxverb']
    conjunction = results['word usage']['conjunction']
    pronoun = results['word usage']['pronoun']
    preposition = results['word usage']['preposition']
    nominalization = results['word usage']['nominalization']
    #6
    pronoun_b = results['sentence beginnings']['pronoun']
    interrogative = results['sentence beginnings']['interrogative']
    article = results['sentence beginnings']['article']
    subordination = results['sentence beginnings']['subordination']
    conjunction_b = results['sentence beginnings']['conjunction']
    preposition_b = results['sentence beginnings']['preposition']
    #6
    
    return [chars_per_word, syll_per_word, words_per_sent,sentences_per_paragraph,word_diversity, type_token_ratio, syllables, words_count, wordtypes, sentences, paragraphs, long_words, complex_words, complex_words_dc, kincaid, ari, coleman_liau, flesch, gunning_fog, lix, smog, rix, dale_chall, tobeverb, auxverb, conjunction, pronoun, preposition, nominalization, pronoun_b, interrogative, article, subordination, conjunction_b, preposition_b]

def spacy_features(df: pd.DataFrame):
    """
    This function generates features using spacy en_core_wb_lg
    I learned about this from these resources:
    https://www.kaggle.com/konradb/linear-baseline-with-cv
    https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
    """
    
    nlp = spacy.load('en_core_web_lg')
    with nlp.disable_pipes():
        vectors = np.array([nlp(text).vector for text in df.excerpt])
        
    return vectors

def get_spacy_col_names():
    names = list()
    for i in range(300):
        names.append(f"spacy_{i}")
        
    return names
def pos_tag_features(passage: str):
    """
    This function counts the number of times different parts of speech occur in an excerpt
    """
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]
    
    tags = pos_tag(word_tokenize(passage))
    tag_list= list()
    
    for tag in pos_tags:
        tag_list.append(len([i[0] for i in tags if i[1] == tag]))
    
    return tag_list

def generate_other_features(passage: str):
    """
    This function is where I test miscellaneous features
    This is experimental
    """
    # punctuation count
    periods = passage.count(".")
    commas = passage.count(",")
    semis = passage.count(";")
    exclaims = passage.count("!")
    questions = passage.count("?")
    qut = passage.count("‘")
    
    # Some other stats
#     num_char = len(passage)
#     num_words = len(passage.split(" "))
#     unique_words = len(set(passage.split(" ") ))
#     word_diversity = unique_words/num_words
    
    word_len = [len(w) for w in passage.split(" ")]
    longest_word = np.max(word_len)
    avg_len_word = np.mean(word_len)
    
    return [periods, commas, semis, exclaims, questions,
            longest_word, avg_len_word]

def create_folds(data: pd.DataFrame, num_splits: int):
    """ 
    This function creates a kfold cross validation system based on this reference: 
    https://www.kaggle.com/abhishek/step-1-create-folds
    """
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data

class CLRDataset:
    """
    This is my CommonLit Readability Dataset.
    By calling the get_df method on an object of this class,
    you will have a fully feature engineered dataframe
    """
    def __init__(self, df: pd.DataFrame, train: bool, n_folds=2):
        self.df = df
        self.excerpts = df["excerpt"]
        
        self._extract_features()
        
        if train:
            self.df = create_folds(self.df, n_folds)
        
    def _extract_features(self):
        scores_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : readability_measurements(p)).tolist(), 
                                 columns=["chars_per_word", "syll_per_word", "words_per_sent","sentences_per_paragraph","word_diversity",
                                          "type_token_ratio","syllables","words_count","wordtypes","sentences","paragraphs","long_words","complex_words","complex_words_dc",
                                          "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog", "rix", "dale_chall",
                                          "tobeverb", "auxverb", "conjunction", "pronoun", "preposition", "nominalization",
                                          "pronoun_b", "interrogative", "article", "subordination", "conjunction_b", "preposition_b"])
        self.df = pd.merge(self.df, scores_df, left_index=True, right_index=True)
        
        spacy_df = pd.DataFrame(spacy_features(self.df), columns=get_spacy_col_names())
        self.df = pd.merge(self.df, spacy_df, left_index=True, right_index=True)
        
        pos_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : pos_tag_features(p)).tolist(),
                              columns=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                                       "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                                       "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"])
        self.df = pd.merge(self.df, pos_df, left_index=True, right_index=True)
        
        other_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : generate_other_features(p)).tolist(),
                                columns=["periods", "commas", "semis", "exclaims", "questions",
                                         "longest_word", "avg_len_word"])
        self.df = pd.merge(self.df, other_df, left_index=True, right_index=True)
        
    def get_df(self):
        return self.df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        pass


features = ["chars_per_word", "syll_per_word", "words_per_sent","sentences_per_paragraph",
            "type_token_ratio","syllables","words_count","wordtypes","sentences","paragraphs",
            "long_words","complex_words","complex_words_dc",
            "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog", "rix", "dale_chall",
            "tobeverb", "auxverb", "conjunction", "pronoun", "preposition", "nominalization",
            "pronoun_b", "interrogative", "article", "subordination", "conjunction_b", "preposition_b"]
features+=get_spacy_col_names()
features+=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
            "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
            "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]

features+= [ "int_count","fer_score","non_fer_count","non_fer_pres","std_sentences_len","std_words_len","mean_of_top10_words_len",
            "mean_of_top2_sentences_len","mean_of_tail2_sentences_len",#"periods","commas", "semis", "exclaims", "questions",
            "word_diversity","longest_word","Kincaid_sen_max","ARI_sen_max","Coleman-Liau_sen_max","FleschReadingEase_sen_max","GunningFogIndex_sen_max","LIX_sen_max","SMOGIndex_sen_max","RIX_sen_max","DaleChallIndex_sen_max"
           ]
# currently better results without the other_df features


def get_frequency_features(df:pd.DataFrame, en_fer_words_path:str=r"../input/english-word-frequency/unigram_freq.csv"):
    en_fer_words = pd.read_csv(en_fer_words_path)
    en_fer_words.index = en_fer_words.word
    en_fer_words.drop("word",axis=1,inplace=True)

    en_fer_words["count"] = en_fer_words["count"]/23135851162 #max
    en_fer_words = en_fer_words.iloc[:60000,:]
    f_w = dict(zip(list(en_fer_words.index) , list(en_fer_words["count"])))


    delimiters = " " , "," , "." , ";" , "!", "\n", "?"
    regexPattern = "|".join(map(re.escape, delimiters))

    df["fer_score"] = 0.0
    df["non_fer_count"] = 0.0
    df["non_fer_pres"] = 0.0
    df["int_count"] = 0.0
    en_fer_words = en_fer_words["count"]
    for j in tqdm(df.index):
        all_words = re.split(regexPattern, re.sub(r'[^\w\s]', '', df.excerpt[j]).replace("\n", " "))
        l = 0
        r = 0
        o = 0
    #     print(all_words)
        for i in all_words:
            fer_read = f_w.get(i.lower(), 0)
            l +=fer_read

            if fer_read == 0:
                try:
                    int(i)
                    o+=1
                except:
                    if i == " " or i == "":
                        continue
                    else:
                        r +=1
        df.loc[j,["fer_score","non_fer_count","non_fer_pres","int_count"]] = l , r , r/len(all_words) , o


    df[["Kincaid_sen_mean","ARI_sen_mean","Coleman-Liau_sen_mean","FleschReadingEase_sen_mean","GunningFogIndex_sen_mean","LIX_sen_mean","SMOGIndex_sen_mean","RIX_sen_mean","DaleChallIndex_sen_mean"]] = 0.0
    df[["Kincaid_sen_std","ARI_sen_std","Coleman-Liau_sen_std","FleschReadingEase_sen_std","GunningFogIndex_sen_std","LIX_sen_std","SMOGIndex_sen_std","RIX_sen_std","DaleChallIndex_sen_std"]] = 0.0
    df[["Kincaid_sen_max","ARI_sen_max","Coleman-Liau_sen_max","FleschReadingEase_sen_max","GunningFogIndex_sen_max","LIX_sen_max","SMOGIndex_sen_max","RIX_sen_max","DaleChallIndex_sen_max"]] = 0.0
    delimiters = " " , "," , "." , ";" , "!", "\n", "?"
    regexPattern = "|".join(map(re.escape, delimiters))

    df["words"] = df.excerpt.apply(lambda x:re.split(regexPattern, x))
    df["std_words_len"] = 0.0
    df["mean_of_top10_words_len"] = 0.0
    for i in tqdm(df.index):
        words_list = df.words[i]
        b = []
        for j in words_list:
            b.append(len(j))
        
        
        result = filter(lambda x: x >1, b)
        b = list(result)
        
        df.std_words_len[i] = np.std(b)
        b.sort()
        df["mean_of_top10_words_len"][i] = np.mean(b[-10:])
        
        

        
    delimiters = "." , "!" , "\n" , ";"  ,"?"
    regexPattern = "|".join(map(re.escape, delimiters))


    df["sentences_list"] = df.excerpt.apply(lambda x:re.split(regexPattern, x))
    df["std_sentences_len"] = 0.0
    df["mean_of_top2_sentences_len"] = 0.0
    df["mean_of_tail2_sentences_len"] = 0.0
    for i in tqdm(df.index):
        sent_list = df.sentences_list[i]
        b = []
        sent_score = []
        
        result = filter(lambda x: len(x) >2, df["sentences_list"][i])
        sent_list = list(result)
        for j in sent_list:
            sent_score.append(list(readability.getmeasures(j, lang='en')['readability grades'].values()))

            b.append(len(j))
            
        sent_score_std = np.std(sent_score,axis=0)
        sent_score_mean = np.mean(sent_score,axis=0) 
        sent_score_max = np.max(sent_score,axis=0) 
        
        
        df.std_sentences_len[i] = np.std(b)
        b.sort()
        df["mean_of_top2_sentences_len"][i] = np.mean(b[-2:])
        df["mean_of_tail2_sentences_len"][i] = np.mean(b[:2])
        df.loc[i,["Kincaid_sen_mean","ARI_sen_mean","Coleman-Liau_sen_mean","FleschReadingEase_sen_mean","GunningFogIndex_sen_mean","LIX_sen_mean","SMOGIndex_sen_mean","RIX_sen_mean","DaleChallIndex_sen_mean"]] = sent_score_mean
        df.loc[i,["Kincaid_sen_std","ARI_sen_std","Coleman-Liau_sen_std","FleschReadingEase_sen_std","GunningFogIndex_sen_std","LIX_sen_std","SMOGIndex_sen_std","RIX_sen_std","DaleChallIndex_sen_std"]] = sent_score_std
        df.loc[i,["Kincaid_sen_max","ARI_sen_max","Coleman-Liau_sen_max","FleschReadingEase_sen_max","GunningFogIndex_sen_max","LIX_sen_max","SMOGIndex_sen_max","RIX_sen_max","DaleChallIndex_sen_max"]] = sent_score_max

    return df