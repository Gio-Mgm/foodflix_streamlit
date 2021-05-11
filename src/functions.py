import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer, models
from streamlit_CONST import STOPS
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process

def fit_model(df, method):
    """
        Fitting chosen model

        params:
            df: DataFrame used,
            method: model chosen

        returns:
            generated model,
            transformed datas
    """

    if method == "TF-IDF":
        model = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                min_df=0, stop_words=STOPS)
        X = model.fit_transform(df['content'])
    elif method == "CountVectorizer":
        model = CountVectorizer(analyzer='word', ngram_range=(1, 2),
                            min_df=0, stop_words=STOPS)
        X = model.fit_transform(df['content'])
    elif method == "BERT":
        word_embedding_model = models.Transformer('camembert-base')
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        X = model.encode(df['content'], show_progress_bar=True)
    return model, X

#=====================================================================#

def find_closest(model, X, query, method):
    """
        Find closest results using cosine similarities

        params:
            model: model used,
            X:

        returns:
            list of indexes of closest results

    """
    if method == "BERT":
        x = model.encode([query])
    else:
        x = model.transform([query])
    cosine_similarities = linear_kernel(x, X)
    similar_indices = cosine_similarities[0].argsort()[:-21:-1]
    return similar_indices

#=====================================================================#

def find_fuzzy(x, series):
    """
        Find with fuzzy wuzzy
    
        params:
            x: target to find,
            series: location to search
    
        returns:
            list of tuples of extractBests
    """
    
    return process.extractBests(x, series, limit=5)

#=====================================================================#

def get_list_of_unique_most(series, thresh=100):
    """
        Get list of most represanted values in a series
    
        params:
            series: series to look at,
            thresh: threshold
        returns:
            list of values
    """
    
    s = pd.Series(", ".join(series).split(','))
    a = s.value_counts()
    return list(a[a > thresh].index[1:])

#=====================================================================#

def get_results(df, found, short):
    results = []

    for i in found:
        name = df.iloc[i].product_name
        brand = df.iloc[i].brands
        nutriscore = df.iloc[i].nutrition_grade_fr
        if short:
            results.append([name, brand, nutriscore])
        else:
            allergens = df.iloc[i].allergens
            ingredients = df.iloc[i].ingredients_text
            vals = df.iloc[i][[
                "energy_100g", "fat_100g", "saturated-fat_100g",
                "carbohydrates_100g", "sugars_100g", "fiber_100g",
                "proteins_100g", "salt_100g"
            ]]
            rename = {
                "energy_100g": "énergie (en kj)",
                "fat_100g": "lipides",
                "saturated-fat_100g": "dont saturés",
                "carbohydrates_100g": "glucides",
                "sugars_100g": "dont sucres",
                "fiber_100g": "fibres",
                "proteins_100g": "protéines",
                "salt_100g": "sel",
            }
            vals = vals.rename(index=rename)
            results.append([name, brand, nutriscore, allergens, ingredients, vals])
    return results
