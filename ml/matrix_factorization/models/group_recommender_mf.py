from turtle import shape
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import pickle

from .explicit_mf_with_bias import SGDExplicitBiasMF

class GroupRecommenderMF(SGDExplicitBiasMF):
    def __init__(self, full_model_file_path, item_encoder_file_path):
        """
        
        """
        with open(full_model_file_path, "rb") as f:
            mf_full_model = pickle.load(file=f)
        self.item_bias = mf_full_model.item_bias
        self.item_vecs = mf_full_model.item_vecs
        self.global_bias = mf_full_model.global_bias
        self.n_factors = mf_full_model.item_vecs.shape[1]
        self.item_encoder_df = pd.read_csv(item_encoder_file_path)
        self.item_encoder_df.rename(columns={
            "Orignal Id":"original_id",
            "Encoded Id":"encoded_id"
            }, inplace=True)
    

    def recommend_group(self, group_rating_df, reg,
        rec_type="virtual_user",
        agg_method="mean", k=10):
        """
        Input: 
            - group_rating_df (3 columns: user_name, item_id, rating)
                Say we have a group of 5 people: different names
            - reg: regularization parameter, range between 1e-5 and 1e2
            - rec_type: "virtual_user" or "combine_recommender"
            - agg_method: "mean" or "min" (least misery) or "max" (most happiness)
        Output:
            - top_k: (2, 10), columns = ["item_id", "rating"]
        """
        self.n_users_in_group = int(group_rating_df["user_name"].nunique())
        group_rating_df_encoded = self.item_encode(group_rating_df)
        virtual_user_rating = self.agg_virtual(group_rating_df_encoded, agg_method)

        # Calculate recommendation for each user
        distinct_users = group_rating_df["user_name"].unique()
        predicted_group = dict()
        for user in distinct_users:
            user_rating = group_rating_df_encoded[group_rating_df_encoded["user_name"] == user]
            user_rating_ = self.agg_virtual(user_rating, agg_method="mean") # Convert to sparse matrix
            p_u, b_u = self.train_virtual(user_rating_, reg)
            predicted_user = self.predict_virtual(p_u, b_u)
            predicted_group[user] = predicted_user
        predicted_group = pd.DataFrame(predicted_group)

        if rec_type == "virtual_user":
            p_g, b_g = self.train_virtual(virtual_user_rating, reg)
            predicted_virtual = self.predict_virtual(p_g, b_g)

        elif rec_type == "combine_recommender":
            predicted_virtual = self.combine_recommender(predicted_group, agg_method)

        df_k_encoded = self.sort_and_filter(predicted_virtual, virtual_user_rating, k=k)
        df_top_k_encoded = df_k_encoded.join(predicted_group, how="left", on="item_id_encoded")
        top_k = self.item_decode(df_top_k_encoded)

        return top_k
    
    def combine_recommender(self, predicted_group, agg_method="mean"):
        if agg_method == "mean":
            predicted_virtual = predicted_group.mean(axis=1)
        elif agg_method == "min":
            predicted_virtual = predicted_group.min(axis=1)
        elif agg_method == "max":
            predicted_virtual = predicted_group.max(axis=1)
        return predicted_virtual.values

    def item_decode(self, df_top_k_encoded):
        """
        Input:
            - df_top_k_encoded: (2, k), columns = ["item_id_encoded", "rating"]
        Output:
            - top_k: (2, k), columns = ["item_id", "rating"]
        """
        top_k = df_top_k_encoded.merge(
            self.item_encoder_df, 
            left_on="item_id_encoded", 
            right_on="encoded_id"
            ).copy()
        top_k.drop(columns=["encoded_id", "item_id_encoded"], inplace=True)
        top_k.rename(columns={
            "original_id":"item_id",
            "rating":"recommendation_score"
            }, inplace=True)
        top_k.sort_values(by="recommendation_score", ascending=False, inplace=True)
        return top_k.reset_index(drop=True)

    def sort_and_filter(self, predicted_virtual, virtual_user_rating, k=10):
        """
        Input:
            - predicted_virtual: (1, n_items_all)
            - virtual_user_rating: Compressed Sparse Row matrix containing the rating 
                shape = (n_users_in_group + 1, n_items_all)
        Output:
            - df_k_encoded: (2, 10), columns = ["item_id_encoded", "rating"]
        """
        predicted_virtual = predicted_virtual.flatten()
        virtual_user_rating = virtual_user_rating.flatten()
        predicted_virtual[virtual_user_rating > 0] = -np.inf
        k_encoded = np.argsort(predicted_virtual)[-k:]
        top_k_rating = predicted_virtual[k_encoded]
        df_k_encoded = pd.DataFrame({
            'item_id_encoded': k_encoded,
            'rating': top_k_rating
        })
        return df_k_encoded

    def predict_virtual(self, virtual_user_embedding, virtual_user_bias):
        """`
        Input:
            - virtual_user_embedding: (1, n_factors)
            - virtual_user_bias: (1,1)
        Output:
            - predicted_virtual: (1, n_items_all)
        """
        predicted_virtual = self.global_bias + virtual_user_bias + \
            self.item_bias + virtual_user_embedding.dot(self.item_vecs.T)
        
        return predicted_virtual

    def train_virtual(self, virtual_user_rating, reg):
        """
        Input:
            - virtual_user_rating: Numpy array containing the rating, shape = (1, num_anime_total)
        """
        n_items_in_group = int((virtual_user_rating > 0).sum()) # num items that's rated by group users
        item_indices = np.argwhere(virtual_user_rating > 0)[:,1].flatten()
        r_ = np.array(virtual_user_rating.T[item_indices])[:,0].flatten()
        b_i_ = self.item_bias[item_indices]
        s_ = r_ - self.global_bias - b_i_
        s_ = s_.reshape(-1, 1)
        A_ = self.item_vecs[item_indices]
        A_ = np.hstack((A_, np.ones((n_items_in_group, 1))))

        pb_g = np.linalg.inv(
            A_.T.dot(A_) + reg * np.eye(N=self.n_factors+1)
            ).dot(A_.T).dot(s_) # (n_factors + 1,1)
        p_g = pb_g[:-1,0].T # (1, n_factors)
        b_g = pb_g[-1,0] # (1,1)
        
        return p_g, b_g

    def agg_virtual(self, group_rating_df_encoded, agg_method="mean"):
        """
        Input: group_rating_df_encoded (3 columns: user_name, item_id_encoded, rating)
        Output: 
            - virtual_user_rating: Numpy array containing the rating, shape = (1, num_anime_total)
        """
        if agg_method == "mean":
            agg_df = group_rating_df_encoded.groupby("item_id")["rating"].mean()
        elif agg_method == "min":
            agg_df = group_rating_df_encoded.groupby("item_id")["rating"].min()
        elif agg_method == "max":
            agg_df = group_rating_df_encoded.groupby("item_id")["rating"].max()
        num_anime_total = self.item_vecs.shape[0]
        virtual_user_rating = np.zeros(shape=(num_anime_total, 1))
        virtual_user_rating[agg_df.index] = agg_df.values.reshape(-1,1)
        virtual_user_rating = virtual_user_rating.T
        return virtual_user_rating
    
    def item_encode(self, group_rating_df):
        """
        Keep consistent with encoding, access encoding dictionary
        """
        group_df = group_rating_df.copy()
        encode_df = self.item_encoder_df.set_index(
            "original_id")["encoded_id"].to_dict()
        group_df["item_id"] = group_df.item_id.apply(encode_df.get)

        return group_df