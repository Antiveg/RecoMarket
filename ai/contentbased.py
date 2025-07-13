import pandas as pd
import os
import re
import pickle
from scipy.sparse import hstack
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text_feature(data : pd.DataFrame, lowercase=True, stopword=True, lemma=False, stem=False, punc=True, number=True, url=False, email=False):

    punctuation = string.punctuation
    stopwordlist = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")

    def remove_urls(text):
        return re.sub(r"http\S+|www\S+|https\S+", '', text)

    def remove_emails(text):
        return re.sub(r'\S+@\S+', '', text)

    def text_cleaning(sentence):
        if not isinstance(sentence, str): return ""
        if lowercase:
            sentence = sentence.lower()
        if url: 
            sentence = remove_urls(sentence)
        if email: 
            sentence = remove_emails(sentence)
        words = word_tokenize(sentence)
        if punc: 
            words = [word for word in words if word not in punctuation]
        if stopword: 
            words = [word for word in words if word not in stopwordlist]
        if number: 
            words = [word for word in words if word.isalpha()]
        if lemma:
            pos_tags = pos_tag(words)
            words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        if stem:
            words = [stemmer.stem(word) for word in words]
        return " ".join(words)

    df = data.copy()
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: text_cleaning(x))
    return df

def encode(dataset : pd.DataFrame, sparse=False):
    data = dataset.copy()
    column = dataset.columns[0]
    model_path=f'./models/{column}_onehotencoder.pkl'
    if os.path.exists(model_path):
        print(f"Loading existing encoder from {model_path}")
        encoder = pd.read_pickle(model_path)
    else:
        print(f"Training a new aisle encoder and saving it to {model_path}")
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)
        encoder.fit(data)
        pd.to_pickle(encoder, model_path)
    dataset_encoded = encoder.transform(data)
    returned = dataset_encoded if sparse else pd.DataFrame(dataset_encoded, columns=encoder.get_feature_names_out([column]), index=dataset.index)
    return returned

def raw_to_tfidf(dataset: pd.DataFrame, sparse=False):
    column = dataset.columns[0]
    data = dataset[[column]].copy()
    model_path = f'./models/{column}_tfidf_vectorizer.pkl'

    if os.path.exists(model_path):
        print(f"Loading existing vectorizer from {model_path}")
        with open(model_path, 'rb') as file:
            vectorizer = pickle.load(file)
    else:
        print(f"Training new TF-IDF vectorizer and saving it to {model_path}")
        vectorizer = TfidfVectorizer()
        vectorizer.fit(data[column])
        with open(model_path, 'wb') as file:
            pickle.dump(vectorizer, file)

    tfidf_matrix = vectorizer.transform(data[column])
    if sparse:
        return tfidf_matrix, vectorizer
    else:
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=vectorizer.get_feature_names_out([column]),
            index=data.index
        ), vectorizer

def extract_features(userproducts : pd.DataFrame):
    tfidf_matrix, _ = raw_to_tfidf(userproducts[['product_name']], sparse=True)
    encoded_aisle = encode(userproducts[['aisle']], sparse=True)
    encoded_department = encode(userproducts[['department']], sparse=True)
    id_map = userproducts[['product_id']]
    return hstack([tfidf_matrix, encoded_aisle, encoded_department]), id_map

def get_userproduct_ids(user_id=17):
    orderproducts = pd.read_csv('./instacart/order_products__train.csv')
    orders = pd.read_csv('./instacart/orders.csv')
    userorders = orders[orders['user_id'].isin([user_id])]['order_id']
    userproducts = set(orderproducts[orderproducts['order_id'].isin(userorders)]['product_id'])
    return userproducts

def get_detailed_products():
    aisles = pd.read_csv('./instacart/aisles.csv')
    departments = pd.read_csv('./instacart/departments.csv')
    products = pd.read_csv('./instacart/products.csv')
    detailed_products = products.merge(aisles, on='aisle_id', how='inner')
    detailed_products = detailed_products.merge(departments, on='department_id', how='inner')
    detailed_products = detailed_products.drop(columns=['aisle_id','department_id'])
    return detailed_products

def train_nn_model(features):
    nn_path='./models/nn_model_production.pkl'
    mapping_path='./models/product_mapping_production.pkl'
    preprocessed_features, product_mapping = extract_features(features)

    model = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    model.fit(preprocessed_features)

    with open(nn_path, 'wb') as f:
        pickle.dump(model, f)
    with open(mapping_path, 'wb') as f:
        pickle.dump(product_mapping, f)

    print(f"KNN model saved to {nn_path}")
    print(f"Product mapping saved to {mapping_path}")
    return model, product_mapping

def load_nn_model_and_mapping():

    if os.path.exists('./models/nn_model_production.pkl') and os.path.exists('./models/product_mapping_production.pkl'):
        with open('./models/nn_model_production.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./models/product_mapping_production.pkl', 'rb') as f:
            product_mapping = pickle.load(f)
    else:
        products = get_detailed_products()
        model, product_mapping = train_nn_model(products)
    return model, product_mapping

def recommendNN(user_id=17, products=None, top_n=50):
    if products == None:
        detailed_products = get_detailed_products()
        userproduct_ids = get_userproduct_ids(user_id)
        products = detailed_products[detailed_products['product_id'].isin(userproduct_ids)]

    model, product_mapping = load_nn_model_and_mapping()
    preprocessed_products, _ = extract_features(products)
    
    tally = defaultdict(list)
    for i in range(preprocessed_products.shape[0]):
        product_vector = preprocessed_products[i].reshape(1, -1)
        distances, indices = model.kneighbors(product_vector, n_neighbors=top_n)
        for dist, idx in zip(distances[0], indices[0]):
            tally[idx].append(dist)

    avg_distances = [(idx, sum(dists) / len(dists)) for idx, dists in tally.items()]
    sorted_indices = sorted(avg_distances, key=lambda x: x[1])[:top_n]
    recommended_ids = [(product_mapping.iloc[idx]['product_id'], score) for idx, score in sorted_indices]
    recommended_ids = [(idx, score) for idx, score in recommended_ids if idx not in userproduct_ids]
    return recommended_ids