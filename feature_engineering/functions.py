import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re



"""
___________________________________

Feature Engineering Functions for 
Artwork Classifier

___________________________________

"""


def text_based_transformer(df, medium='Medium', dimensions='Dimensions', title='Title'):
    # Add features based on Medium description
    
    for feature, func in { 'long_list_of_things' : lambda x: long_list_of_things(str(x))
                          ,'installation_keyword' : lambda x: installation_keyword(str(x))
                          ,'print_keyword' : lambda x: print_keyword(str(x))
                          ,'drawing_paper_word' : lambda x: drawing_paper_word(str(x))
                          ,'drawing_keyword' : lambda x: drawing_keyword(str(x))
                          ,'painting_keyword' : lambda x: painting_keyword(str(x))
                          ,'photo_keyword' : lambda x: photo_keyword(str(x))
                              }.items():

        df[feature] = df[medium].apply(func)

    # Add features based on dimensions
    df['has_3dimensions'] = df[dimensions].apply(lambda x: has_3dimensions(str(x)))

    # Add features based on title/description
    for feature, func in {'furniture_keyword' : lambda x: furniture_keyword(str(x))
                          ,'decorative_obj_keyword' : lambda x: decorative_obj_keyword(str(x))
                         }.items():
        
        df[feature] = df[title].apply(func)
        
    return df


def has_3dimensions(x):
    string = str(x).replace(" ", "").replace(".", "").replace("/", "").replace("'", "").lower()
    try:
        re.search(r'\d+x\d+x\d+', string).group()
        return True
    except AttributeError:
        for term in ['dia', 'high', 'tall', 'long']:
            s = string.find(term)
            if s != -1:
                return True
    return False


def painting_keyword(x):
    
    # Manual segregation by picking out key words in the title
    
    painting_word_list = ['on canvas'
                      , 'on wood'
                      , 'oil on'
                      , 'acrylic'
                      , 'tempera'
                      , 'gouache'
                      , ' board'
                      , 'painted'
                      , 'watercolo'
                      , 'mixed media'
                      , 'collage']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in painting_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def photo_keyword(x):
    
    # Manual segregation by picking out key words in the title
    
    photo_word_list = ['silver print'
                         , 'gelatin print'
                         , 'gelatin silver'
                         , 'selenium tone'
                         , 'platinum print'
                         , 'negative'
                         , 'instant print'
                         , 'polaroid'
                         , 'cibachrome'
                         , 'chromogenic'
                         ,  'photo'
                         , 'inkjet print'
                         , 'vibrachrome'
                         , 'selenium tone']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in photo_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def drawing_keyword(x):
    
    # Manual segregation by picking out key words in the medium
    
    drawing_word_list = ['pencil'
                         , 'charcoal'
                         , 'ink'
                         , 'pen'
                         , 'crayon'
                         , 'pastel'
                         , 'chalk'
                         , 'carbon paper transfer'
                         , 'graphite']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in drawing_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def drawing_paper_word(x):
    
    # Manual segregation by picking out key words in the medium
    
    drawing_paper_word_list = ['on paper', 'colored paper', 'graph paper']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in drawing_paper_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def print_keyword(x):
    
    # Manual segregation by picking out key words in the medium
    
    print_word_list = ['lithograp'
                         , 'etching'
                         , 'letterpress'
                         , 'screenprint'
                         , 'woodcut'
                         , 'reproduction'
                         , 'aquatint'
                         , 'drypoint'
                         , 'blueprint'
                         , 'pochoir'
                         , 'intaglio'
                         , 'engraving'
                         , 'carborundum print'
                         , 'mezzotints'
                         , 'diazotype'
                         , 'photocopy'
                         , 'linoleum'
                         , 'linocut'
                         , 'emboss'
                         , 'monotype']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in print_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def installation_keyword(x):
    
    # Manual segregation by picking out key words in the medium
    
    install_word_list = ['installation'
                         , 'on wall'
                         , 'audio'
                         , 'TV' 
                         , 'projector'
                         , 'projection screen'
                         , 'sound system'
                         , 'video'
                         , 'computer'
                         , 'pneumatic'
                         , 'display case'
                         , 'helium'
                         , ' parts'
                         , 'lamp'
                         , 'light'
                         , 'wire']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in install_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def furniture_keyword(x):
    
    # Manual segregation by picking out key words in the medium
    
    furniture_word_list = ['table', 'chair', 'rocker', 'stool']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in furniture_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def sculpture_keyword(x):
    
    # Manual segregation by picking out key words in the medium
    
    sculpture_word_list = ['bronze', 'patina']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in sculpture_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
    return False
    


def decorative_obj_keyword(x):
    
    # Manual segregation by picking out key words in the medium
    
    dec_obj_word_list = ['vase', 'dish', 'tumblers', 'bowl','desk fan'
                         , 'table fan', 'flatware', 'dinnerwear','pitcher'
                         , 'goblet', 'lamp', 'plate', 'clock', 'wall'
                          'plate', 'plaque']

    list_words_cleaned = re.sub(r'\W+', ' ', str(x).lower())

    for x in dec_obj_word_list:
            if list_words_cleaned.find(x) != -1:
                return True
            
    return False

def long_list_of_things(x):
    if len(re.findall(',', x)) >= 4:
        return True
    else:
        return False
    
def area(df, height_col, width_col):
    
    # Need to provide height & width columns (float) in centimeters
    
    df['area'] = df[height_col]*df[width_col]
    df['area'] = df['area'].apply(lambda x: x if x > 1 else 0)
    return df


def is_drawing(df):
    if ('drawing_keyword' in df.columns) and ('drawing_keyword' in df.columns):
        df['is_drawing'] = df['drawing_keyword'] + df['drawing_paper_word']
        df['is_drawing'] = df['is_drawing'].apply(lambda x: False if x == False else True)
        return df
    else:
        print( 'is_drawing column not added, please run drawing_keyword & drawing_paper_word first')
        return df