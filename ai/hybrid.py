import numpy as np
from sklearn.preprocessing import MinMaxScaler
from contentbased import recommendNN 
from collaborative import recommendSVD

def hybrid_recommendation(user_id=17, top_n=10, weight_svd=0.5, weight_nn=0.5):
    original_svd = recommendSVD(user_id=user_id, top_n=25)
    original_nn = recommendNN(user_id=user_id, top_n=25)

    svd_dict = dict(original_svd)
    nn_dict = dict(original_nn)
    all_items = list(set(svd_dict.keys()).union(nn_dict.keys()))

    svd_scores = [svd_dict.get(item, 0.0) for item in all_items]
    nn_scores = [1.0 - nn_dict.get(item, 1.0) for item in all_items]

    svd_scaled = MinMaxScaler().fit_transform(np.array(svd_scores).reshape(-1, 1)).flatten()
    nn_scaled = MinMaxScaler().fit_transform(np.array(nn_scores).reshape(-1, 1)).flatten()
    combined_scores = weight_svd * svd_scaled + weight_nn * nn_scaled

    hybrid_recommendations = list(zip(all_items, combined_scores))
    hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)
    return hybrid_recommendations[:top_n]