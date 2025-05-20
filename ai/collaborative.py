import pickle
import os
import pandas as pd
import pickle
import os
from surprise import SVDpp, Dataset, Reader

def make_model_prod():
    model_filename = './Collaborative/model_surprise_production.pkl'
    trainset_filename = './Collaborative/trainset_surprise_production.pkl'
    
    if os.path.exists(model_filename) and os.path.exists(trainset_filename):
        print(f"Loading model from {model_filename}...")
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        with open(trainset_filename, 'rb') as f:
            trainset = pickle.load(f)
        print("Model loaded successfully!")
        return trainset, model
    
    print('Reading and merging csv dataset ...')
    orderproducts = pd.read_csv('./instacart/order_products__train.csv')
    orders = pd.read_csv('./instacart/orders.csv')
    orders = orders[orders['eval_set'] == 'train']
    userproducts = orderproducts.merge(orders, on='order_id', how='inner')
    reader = Reader(rating_scale=(0, 1))
    trainset = Dataset.load_from_df(userproducts[['user_id','product_id','reordered']], reader).build_full_trainset()

    print('Training Model SVD++ ...')
    model = SVDpp(
        n_factors=20,
        n_epochs=20,
        cache_ratings=False,
        init_mean=0,
        init_std_dev=0.1,
        lr_all=0.007,
        reg_all=0.02,
        lr_bu=0.007,
        lr_bi=0.007,
        lr_pu=0.007,
        lr_qi=0.007,
        lr_yj=0.007,
        reg_bu=0.02,
        reg_bi=0.02,
        reg_pu=0.02,
        reg_qi=0.02,
        reg_yj=0.02,
        random_state=42,
        verbose=True
    )
    model.fit(trainset)

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    with open(trainset_filename, 'wb') as f:
        pickle.dump(trainset, f)
    return model, trainset

def load_model_and_data():
    model_filename = './Collaborative/model_surprise_production.pkl'
    trainset_filename = './Collaborative/trainset_surprise_production.pkl'

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        with open(trainset_filename, 'rb') as f:
            trainset = pickle.load(f)
    else:
        model, trainset = make_model_prod()
    return model, trainset
    
def recommendSVD(user_id=17, top_n=100):

    model, trainset = load_model_and_data()

    all_products = set(trainset.all_items())
    interacted_products = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]])
    candidates = list(all_products - interacted_products)

    predictions = [model.predict(user_id, trainset.to_raw_iid(iid)) for iid in candidates]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:top_n]
    recommended = [(pred.iid, pred.est) for pred in top_predictions]
    return recommended