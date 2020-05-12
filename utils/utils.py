
from IPython.display import display, Image
from IPython.core.display import HTML


"""
___________________________________

Feature Engineering Functions for 
Artwork Classifier

___________________________________

"""


def view_images_from_url(df, url_col, subsample=True, sample_size=20, random_state=0):
    
    # Take a dataframe a displays images from a URL, subsampling optional
    
    if subsample == True:
        df_dict = df[url_col].sample(sample_size, random_state=random_state).to_dict()
    else:
        df_dict = df[url_col].to_dict()
        
    for key, value in df_dict.items():
        print (key)
        display(Image(url= value, width=250, height=250))