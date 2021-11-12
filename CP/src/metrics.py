import numpy as np


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    try:
        recommended_list = recommended_list[:k]
    except:
        print(f'recommended: {recommended_list}, type: {type(recommended_list)}')
        return 0
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        if flags[i-1] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result


def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list[:k])
    
    hit_rate = (flags.sum() > 0) * 1 
    
    return hit_rate

    
def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    
    bought_list = bought_list
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    precision = flags[:k].dot(prices_recommended) / prices_recommended.sum()
     
    return precision


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)
    
    if k < len(recommended_list):
        recommended_list = recommended_list[:k]
    if k < len(prices_recommended):
        prices_recommended = prices_recommended[:k]

    flags = np.isin(recommended_list, bought_list)
    if k < len(flags):
        flags = flags[:k]
    
    recall = flags.dot(prices_recommended) / prices_bought.sum()
    
    return recall


def ndcg_k(recommended_list, bought_list, k=5, discount_func=np.log2):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    discount_list = np.array(list(map(lambda x: 1/discount_func(x+1), np.arange(1, k+1))))
    result = np.isin(recommended_list[:k], bought_list).dot(discount_list) / discount_list.sum()
    
    return result