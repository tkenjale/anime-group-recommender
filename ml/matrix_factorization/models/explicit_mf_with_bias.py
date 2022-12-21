import numpy as np
from sklearn.metrics import mean_squared_error
import time
import pickle
from copy import deepcopy

class SGDExplicitBiasMF:
    def __init__(self, 
                 ratings,
                 ratings_eval,
                 n_factors=64,
                 early_stopping_rounds=10,
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=False,
                 model_saving_path="."):
        """
        Link: [ExplicitMF](https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/)
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        
        item_fact_reg : (float)
            Regularization term for item latent factors
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            
        item_bias_reg : (float)
            Regularization term for item biases
        
        user_bias_reg : (float)
            Regularization term for user biases
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.ratings_eval = ratings_eval
        self.n_factors = n_factors
        self.early_stopping_rounds = early_stopping_rounds
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)
        self._v = verbose
        self._manual_init_bias = False
        self.model_saving_path = model_saving_path

    def init_bias(self, user_bias_init, item_bias_init):
        self.global_bias = np.mean(self.ratings[self.ratings != 0])
        self.user_bias = user_bias_init
        self.item_bias = item_bias_init
        self._manual_init_bias = True

    def train(self, max_iter=200, learning_rate=0.005, pretrained=False):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        if not pretrained:
            self.user_vecs = np.random.normal(scale=1./self.n_factors,\
                                            size=(self.n_users, self.n_factors))
            self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                            size=(self.n_items, self.n_factors))
        
        self.learning_rate = learning_rate
        if (not self._manual_init_bias) and (not pretrained):
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[self.ratings != 0])

        self.min_mse_eval = np.Inf
        self.list_mse_eval = []

        self.partial_train(n_iter=max_iter, save_interim = True)
    
    
    def partial_train(self, n_iter, save_interim = True):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        iter_cnt = 1
        while iter_cnt <= n_iter:
            
            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)
            self.iter_idx = iter_cnt
            self.sgd()
            # Save interim model
            if save_interim:
                now = int(time.time())
                to_save = deepcopy(self)
                to_save.ratings = np.zeros((2,2))
                to_save.ratings_eval = np.zeros((2,2))
                with open(f"{self.model_saving_path}/model_sgd_mf_v4_{self.iter_idx}__{now}.pkl", "wb") as f:
                    pickle.dump(to_save, file=f)
            # evaluate the model
            eval_mse = self.evaluate(self.ratings_eval)
            self.min_mse_eval = min(eval_mse, self.min_mse_eval)
            self.list_mse_eval.append(eval_mse)
            if self._v:
                print(f"Iteration {iter_cnt}. Latest MSE: {eval_mse:.4f}. Min MSE: {self.min_mse_eval:.4f}.")
            

            if min(self.list_mse_eval[-self.early_stopping_rounds:]) > self.min_mse_eval: 
                print("Early stopping due to non-improvement on the test set")
                break
            iter_cnt += 1

    def load_model(self, pickled_file, rating_train, rating_eval):
        with open(pickled_file, "rb") as f:
            self = pickle.load(file=f)
            self.ratings = rating_train
            self.ratings_eval = rating_eval

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u,i] - prediction) # error
            
            # Update biases
            self.user_bias[u] += self.learning_rate * \
                                (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                (e - self.item_bias_reg * self.item_bias[i])
            
            #Update latent factors
            self.user_vecs[u, :] += self.learning_rate * \
                                    (e * self.item_vecs[i, :] - \
                                     self.user_fact_reg * self.user_vecs[u,:])
            self.item_vecs[i, :] += self.learning_rate * \
                                    (e * self.user_vecs[u, :] - \
                                     self.item_fact_reg * self.item_vecs[i,:])
            # if idx % 1000 == 0:
            #     print(idx)
            
    def predict(self, u, i):
        """ Single user and item prediction."""
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
        return predictions

    def evaluate(self, test_sparse_matrix):
        nz_row, nz_col = test_sparse_matrix.nonzero()
        n_idx = len(nz_row)
        rating_pred = np.zeros(n_idx)
        rating_true = np.zeros(n_idx)
        for idx in np.arange(n_idx):
            irow, icol = nz_row[idx], nz_col[idx]
            rating_pred[idx] = self.predict(irow, icol)
            rating_true[idx] = test_sparse_matrix[irow, icol]
        mse = mean_squared_error(rating_true, rating_pred)
        return mse
        
    
