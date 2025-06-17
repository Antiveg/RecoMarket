# import pickle
# import os
# import pandas as pd
# import pickle
# import os
# from surprise import SVDpp, Dataset, Reader

# def make_model_prod():
#     model_filename = './Collaborative/model_surprise_production.pkl'
#     trainset_filename = './Collaborative/trainset_surprise_production.pkl'
    
#     if os.path.exists(model_filename) and os.path.exists(trainset_filename):
#         print(f"Loading model from {model_filename}...")
#         with open(model_filename, 'rb') as f:
#             model = pickle.load(f)
#         with open(trainset_filename, 'rb') as f:
#             trainset = pickle.load(f)
#         print("Model loaded successfully!")
#         return trainset, model
    
#     print('Reading and merging csv dataset ...')
#     orderproducts = pd.read_csv('./instacart/order_products__train.csv')
#     orders = pd.read_csv('./instacart/orders.csv')
#     orders = orders[orders['eval_set'] == 'train']
#     userproducts = orderproducts.merge(orders, on='order_id', how='inner')
#     reader = Reader(rating_scale=(0, 1))
#     trainset = Dataset.load_from_df(userproducts[['user_id','product_id','reordered']], reader).build_full_trainset()

#     print('Training Model SVD++ ...')
#     model = SVDpp(
#         n_factors=20,
#         n_epochs=20,
#         cache_ratings=False,
#         init_mean=0,
#         init_std_dev=0.1,
#         lr_all=0.007,
#         reg_all=0.02,
#         lr_bu=0.007,
#         lr_bi=0.007,
#         lr_pu=0.007,
#         lr_qi=0.007,
#         lr_yj=0.007,
#         reg_bu=0.02,
#         reg_bi=0.02,
#         reg_pu=0.02,
#         reg_qi=0.02,
#         reg_yj=0.02,
#         random_state=42,
#         verbose=True
#     )
#     model.fit(trainset)

#     with open(model_filename, 'wb') as file:
#         pickle.dump(model, file)
#     with open(trainset_filename, 'wb') as f:
#         pickle.dump(trainset, f)
#     return model, trainset

# def load_model_and_data():
#     model_filename = './Collaborative/model_surprise_production.pkl'
#     trainset_filename = './Collaborative/trainset_surprise_production.pkl'

#     if os.path.exists(model_filename):
#         with open(model_filename, 'rb') as f:
#             model = pickle.load(f)
#         with open(trainset_filename, 'rb') as f:
#             trainset = pickle.load(f)
#     else:
#         model, trainset = make_model_prod()
#     return model, trainset

import pandas as pd
from typing import Tuple

import pickle as pkl
import os

from surprise import SVDpp, Dataset, Reader, Trainset

def prepare_csv_dataset():
    print('(Data Cleaning 0/5) Reading and merging csv dataset ...')
    orderproducts = pd.read_csv('./instacart/order_products__train.csv')
    orders = pd.read_csv('./instacart/orders.csv')

    print('(Data Cleaning 1/5) Filter to only include orders which is in train and test set...')
    orders = orders[orders['eval_set'].isin(['train'])]

    print('(Data Cleaning 2/5) Merge orderproducts and orders, then remove irrelevant columns...')
    userproducts = orderproducts.merge(orders, on='order_id', how='inner')
    userproducts = userproducts[['product_id','user_id','reordered']]

    print('(Data Cleaning 3/5) Remove user which only occur once to prevent error during testing...')
    uid_counts = userproducts['user_id'].value_counts()
    valid_uids = uid_counts[uid_counts > 1].index
    userproducts = userproducts[userproducts['user_id'].isin(valid_uids)]

    print('(Data Cleaning 4/5) Remove duplicated data and missing value if exists...')
    userproducts = userproducts.dropna(inplace=False)
    userproducts = userproducts.drop_duplicates(inplace=False)

    print('(Data Cleaning 5/5) Data Cleaning process is done!')
    return userproducts

def preprocess_dataset(data) -> Tuple[Trainset, list | None]:
    userproducts = data.copy()
    print('(Data Preprocessing 0/1) Convert dataset into surprise specific dataset...')
    reader = Reader(rating_scale=(0, 1))
    print(len(userproducts))
    print(userproducts)
    trainset = Dataset.load_from_df(userproducts[['user_id','product_id','reordered']], reader).build_full_trainset()

    print('(Data Preprocessing 1/1) Data Preprocessing process is done!')
    return trainset, None
    
def train_model(trainset : Trainset):
    print('(Model Training 0/1) Model SVDpp training in progress ...')
    model = SVDpp(
        n_factors=20,           n_epochs=20,
        cache_ratings=False,    init_mean=0,
        init_std_dev=0.1,       lr_all=0.007,
        reg_all=0.02,           lr_bu=0.007,
        lr_bi=0.007,            lr_pu=0.007,
        lr_qi=0.007,            lr_yj=0.007,
        reg_bu=0.02,            reg_bi=0.02,
        reg_pu=0.02,            reg_qi=0.02,
        reg_yj=0.02,            random_state=42,
        verbose=True
    )
    model.fit(trainset)
    print('(Model Training 1/1) Model SVDpp training finished ...')
    return model

def load_model(state="production") -> Tuple[SVDpp, Trainset]:
    
    model_filename = f'./models/svd_model_{state}.pkl'
    trainset_filename = f'./models/svd_trainset_{state}.pkl'

    if os.path.exists(model_filename) and os.path.exists(trainset_filename):
        print('Model already exist, currently load it ...')
        with open(model_filename,'rb') as f:
            model = pkl.load(f)
        with open(trainset_filename,'rb') as f:
            trainset = pkl.load(f)

        print('Model successfully loaded ...')
        return model, trainset
    else:
        print('Model doesnt exist yet. Train new model ...')
        cleaned_userproducts = prepare_csv_dataset()
        trainset, _ = preprocess_dataset(cleaned_userproducts)
        model = train_model(trainset)

        print('Saving trained model ...')
        with open(model_filename,'wb') as f:
            pkl.dump(model, f)
        with open(trainset_filename,'wb') as f:
            pkl.dump(trainset, f)

        print('Model successfully saved ...')
        return model, trainset
    
def recommendSVD(user_id=17, top_n=100):
    model, trainset = load_model()

    all_products = set(trainset.all_items())
    interacted_products = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]])
    candidates = list(all_products - interacted_products)

    predictions = [model.predict(user_id, trainset.to_raw_iid(iid)) for iid in candidates]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:top_n]
    recommended = [(pred.iid, pred.est) for pred in top_predictions]

    return recommended