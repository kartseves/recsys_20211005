import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting='bm25_weight', n_factors=20):
        
        # Общий топ покупок
        self.items_top = data.groupby('item_id')['quantity'].count().reset_index()
        self.items_top.sort_values('quantity', ascending=False, inplace=True)
        self.items_top = self.items_top[self.items_top['item_id'] != 999999]
        self.items_top = self.items_top.item_id.tolist()

        # Топ покупок пользователей
        self.users_top = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.users_top.sort_values('quantity', ascending=False, inplace=True)
        self.users_top = self.users_top[self.users_top['item_id'] != 999999]

        self.user_item_matrix = self.prepare_matrix(data)
        
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
                
        if weighting=='bm25_weight':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
        elif weighting=='tfidf_weight':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T
        
        self.model = self.fit(self.user_item_matrix, n_factors=n_factors)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        """Подготавливает user-item матрицу"""
        
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает словари мапинга идентификаторов"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных пользователем"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def _update_user_dict(self, user_id):
        """Добавляет маппинг для нового пользователя"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
            
    def _update_item_dict(self, item_id):
        """Добавляет маппинг для нового товара"""

        if item_id not in self.itemid_to_id.keys():
            max_id = max(list(self.itemid_to_id.values()))
            max_id += 1

            self.itemid_to_id.update({item_id: max_id})
            self.id_to_itemid.update({max_id: item_id})
            
    def _get_similar_item(self, item_id):
        """Возвращает item_id товара схожего с переданным товаром"""
        
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        return self.id_to_itemid[recs[1][0] if len(recs) > 1 else recs[0][0]]

    def _extend_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.items_top[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        
        self._update_user_dict(user_id=user)
        
        recs = model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)
        
        res = [self.id_to_itemid[rec[0]] for rec in recs]
        res = self._extend_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
        
    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_user_dict(user_id=user)
        try:
            return self._get_recommendations(user, model=self.model, N=N)
        except:
            print(f'Get recommendations error. Return top. User ID: {user}')
            return self._extend_top_popular([], N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_user_dict(user_id=user)
        try:
            return self._get_recommendations(user, model=self.own_recommender, N=N)
        except:
            print(f'Get recommendations error. Return top. User ID: {user}')
            return self._extend_top_popular([], N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        try:
            top_user_items = self.users_top[self.users_top['user_id'] == user].head(N)

            res = top_user_items['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
            res = self._extend_top_popular(res, N=N)

            assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
            return res
        except:
            print(f'Get recommendations error. Return top. User ID: {user}')
            return self._extend_top_popular([], N=N)

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        try:
            similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
            similar_users = [rec[0] for rec in similar_users]
            similar_users = similar_users[1:]

            res = []
            for user in similar_users:
                res.extend(self.get_own_recommendations(self.id_to_userid[user], N=1))
            res = self._extend_top_popular(res, N=N)

            assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
            return res
        except:
            print(f'Get recommendations error. Return top. User ID: {user}')
            return self._extend_top_popular([], N=N)
