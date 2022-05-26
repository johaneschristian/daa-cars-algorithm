import numpy as np
import pandas as pd
from sklearn import preprocessing
from itertools import product as cartesian_product




def training():
    LMAX = 6      # Maximum transitive path length
    df = pd.read_csv('paper_rating.csv')
    df.tail(5)

    maximum_rating = df[df.columns[2]].max()

    user_item_context_encodings, user_item_context_reverse_encodings, encoded_df = encode_context(df)

    rating_matrix, context_T = item_splitting(user_item_context_encodings, encoded_df, maximum_rating)

    M = user_based_graph_similarity(rating_matrix, LMAX)

    return user_item_context_encodings, user_item_context_reverse_encodings, encoded_df


def prediction(M, user_item_context_encodings, encoded_df, K_similar_users, rating_matrix, user_item_context_reverse_encodings, context_T):
    K = 2
    USER = 'u4'   # User to check
    CONTEXT = ('raining', 'evening', 'weekend')
    N = 10

    translated_context = []

    for cnt_index in range(len(CONTEXT)):
        # 0 --> User
        # 1 --> Item
        # >= 2 --> context
        map_index = cnt_index + 2
        translation_table = user_item_context_encodings[map_index]
        
        translated_context.append(translation_table[CONTEXT[cnt_index]])

    translated_context = tuple(translated_context)

    # Get K most similar users
    wanted_user = user_item_context_encodings[0][USER]

    scores = M[wanted_user]
    K_similar_users = np.argpartition(scores, -(K+1))[-(K+1):]

    if wanted_user in K_similar_users:
        K_similar_users = K_similar_users[K_similar_users != wanted_user]
    else:
        K_similar_users = K_similar_users[:-1]

def KNN(encoded_df, K_similar_users, rating_matrix, user_item_context_reverse_encodings, context_T, wanted_user):
    
    inferred_ratings = []
    user_rated_items = rating_matrix[wanted_user]
    for item in range(len(user_rated_items)):
        rating = user_rated_items[item]
        
        # Item has not been rated
        if rating == 0:
            rating_sum = 0
            neighbor_count = 0
            
            for neighbor in K_similar_users:
                neighbor_rating = rating_matrix[neighbor][item]
                
                # If neighbor has rated the item
                if neighbor_rating != 0:
                    rating_sum += neighbor_rating
                    neighbor_count += 1
                
            inferred_rating = rating_sum / neighbor_count if neighbor_count != 0 else 0
            
            if inferred_rating != 0:
                inferred_ratings.append((item, inferred_rating))


    mapped_ratings = []
    for i in inferred_ratings:
        
        # Get actual item encoding (from the cartesian product result)
        item_context = context_T[i[0]]
        
        item = user_item_context_reverse_encodings[1][item_context[0]]
        
        new_entry = [item]
        
        for j in range(1, len(item_context)):
            new_entry.append(item_context[j])
        
        new_entry.append(i[1])
        
        mapped_ratings.append(new_entry)

    predicted_df = pd.DataFrame(
    mapped_ratings, 
    columns=[
        'Item', 
        *encoded_df.columns[3:], 
        'predicted_rating']
        ).sort_values(by='predicted_rating', ascending=False)

def recommendation_generation(N, predicted_df, translated_context):
    # Select N items to be recommended
    chosen = []

    for data in predicted_df.iterrows():
        data_tup = tuple(data[1])
        
        if tuple(data[1][1:-1]) == translated_context:
            
            if len(chosen) <= N:
                chosen.append(data_tup)

    for i in chosen:
        print(i)

def rating_prediction_translated(inferred_ratings, context_T, user_item_context_reverse_encodings, encoded_df):
    mapped_ratings = []
    for i in inferred_ratings:
        
        # Get actual item encoding (from the cartesian product result)
        item_context = context_T[i[0]]
        
        item = user_item_context_reverse_encodings[1][item_context[0]]
        
        new_entry = [item]
        
        # Translate remaining context
        for j in range(1, len(item_context)):
            # 1 --> item
            # >= 2 --> context
            translator = user_item_context_reverse_encodings[j+1]
            new_entry.append(translator[item_context[j]])
        
        new_entry.append(i[1])
        
        mapped_ratings.append(new_entry)

    pd.DataFrame(
    mapped_ratings, 
    columns=[
        'Item', 
        *encoded_df.columns[3:], 
        'predicted_rating']
    ).sort_values(by='predicted_rating', ascending=False)

def encode_context(df):
    encoder = preprocessing.LabelEncoder()

    user_item_context_encodings = []
    user_item_context_reverse_encodings = []
    

    encoded_df = df.copy()
    # Encode userid, itemid, and contextual informations for item splitting
    for column_index in range(len(df.columns)):
        
        # Column attribute is not rating
        if column_index != 2:
            
            # Fit encoder
            encoder.fit(df[df.columns[column_index]])
            encoded_df[df.columns[column_index]] = encoder.transform(
                    df[df.columns[column_index]]
                )
        
        # Column is nor user or rating
        if column_index != 2:
                user_item_context_encodings.append(
                    dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                )
                user_item_context_reverse_encodings.append(
                    dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
                )
    print(encoded_df)

    return user_item_context_encodings, user_item_context_reverse_encodings, encoded_df

def item_splitting(user_item_context_encodings, encoded_df, maximum_rating):
    # Cartesian product all items and contexts

    users = user_item_context_encodings[0].values()
    items = user_item_context_encodings[1].values()
    contexts = [
        context_trans.values() for context_trans in user_item_context_encodings[2:]
    ]

    context_T = list(cartesian_product(items, *contexts))
    
    # Generate new user-item matrix for new items
    rating_matrix = np.zeros((len(users), len(context_T)), dtype=object)   

    for row in encoded_df.iterrows():
        data = tuple(row[1])
        user = data[0]
        item = data[1]
        rating = data[2]
        context_item = (item, *data[3:])
        
        index = context_T.index(context_item)
        
        rating_matrix[user][index] = rating/maximum_rating 
    
    print(rating_matrix)

    return rating_matrix, context_T

def user_based_graph_similarity(rating_matrix, LMAX):
    rating_matrix_transposed = np.transpose(rating_matrix)
    L = 2

    WWT = np.matmul(rating_matrix, rating_matrix_transposed)
    M = np.matmul(rating_matrix, rating_matrix_transposed)

    while L != LMAX:
        M = np.matmul(WWT, M)
        L = L + 2

    print(M)
    return M



if __name__ == 'main':
    pass
