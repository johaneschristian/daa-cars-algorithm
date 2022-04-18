import csv
import heapq
import time
from itertools import product as cartesian_product

def pre_process(file_name):
    users = set()
    items = set()
    rating_values = set()
    contexts = []

    with open(file_name) as file:
        data = list(csv.reader(file))

        # Initialize context set
        context_count = len(data[0][3:])

        for i in range(context_count):
            contexts.append(set())

        for row_index in range(1, len(data)):
            row = data[row_index]
            users.add(row[0])
            items.add(row[1])
            rating_values.add(int(row[2]))

            # Add values to contexts
            for context_index in range(len(contexts)):
                if row[context_index+3] != 'NA':
                    contexts[context_index].add(row[context_index+3])
    
    return data, list(users), list(items), contexts, max(rating_values)


def split_items(data: list, users: list, items: list, contexts: list, max_rating) -> tuple:
    # Produce new context C, a result of cartesian product of the previous contexts
    # Produce new context T, a result of items x context_C
    context_T = list(cartesian_product(items, *contexts))
    rating_matrix = generate_matrix(len(users), len(context_T), default=None)

    for i in range(1, len(data)):
        row = data[i]
        user = row[0]
        rating = row[2]
        item = (row[1], *row[3:])

        if not 'NA' in item:
            index_of_user = users.index(user)
            index_of_item = context_T.index(item)

            rating_matrix[index_of_user][index_of_item] = int(rating)/max_rating
    
    return rating_matrix, context_T


# @param current_user
# @param similarity_score_matrix
# @param rating_matrix
# @param k=number of similar users to consider
# @return index of item node to be recommended
def generate_recommendation(current_user: int, similarity_score_matrix: list, rating_matrix: list, k: int, convert_item_index: list, context: list) -> int:
    # Find k users with highest similarity to user_node_index
    scores = similarity_score_matrix[current_user]

    similar_users = heapq.nlargest(k+1, range(len(scores)), key=lambda index: scores[index])

    # Remove current_user from similar_users (if any), else remove the last element
    if similar_users.count(current_user) > 0:
        index = similar_users.index(current_user)
        similar_users.pop(index)
    else:
        similar_users.pop()
    
    user_rating_matrix = rating_matrix[current_user]

    inferred_ratings = []

    # generate rating for nonrated items
    for item_index in range(len(user_rating_matrix)):
        if user_rating_matrix[item_index] == None:
            sum_of_rating = 0
            neighbor_count = 0

            # Infer rating from neighbors
            for neighbor_index in range(len(similar_users)):
                neighbor_rating = rating_matrix[neighbor_index][item_index]

                if neighbor_rating:
                    sum_of_rating += neighbor_rating
                    neighbor_count += 1
            
            average_rating = sum_of_rating/neighbor_count if sum_of_rating else 0
                
            inferred_ratings.append((item_index, average_rating))
    
    recommended_item_index = None
    recommended_item_rating = 0

    # Context takes the format of (context 1, context 2, ...)    
    # Item index takes the format of (product, context 1, context 2, ...)
    for i in range(len(inferred_ratings)):

        item_index = inferred_ratings[i][0]
        item = convert_item_index[item_index]

        # Find matching item
        if not recommended_item_index and item[1:] == context:
            recommended_item_index = item_index
            recommended_item_rating = inferred_ratings[i][1]
        elif item[1:] == context and inferred_ratings[i][1] > recommended_item_rating:
            recommended_item_index = item_index
            recommended_item_rating = inferred_ratings[i][1]
    
    return recommended_item_index

        
# @param rating_matrix rm[i][j] contains the rating
#        given by user at node i to item at node j
# @return adjacency matrix of size |U|+|T| x |U|+|T|
def construct_graph_adjacency_matrix(rating_matrix: list) -> list:
    number_of_users = len(rating_matrix)
    number_of_items = len(rating_matrix[0])

    # Initialize adjacency matrix with 0s
    adjacency_matrix = generate_matrix(number_of_users+number_of_items, number_of_users+number_of_items, default=0)

    # Fill the adjacency matrixZ
    for i in range(number_of_users):
        for j in range(number_of_items):
            if rating_matrix[i][j] == None:
                adjacency_matrix[i][number_of_users+j] = 0
                adjacency_matrix[i+number_of_users][j] = 0
            else:
                adjacency_matrix[i][number_of_users+j] = rating_matrix[i][j]
                adjacency_matrix[number_of_users+j][i] = rating_matrix[i][j]

    return adjacency_matrix


# @param adjacency_matrix: 2D list of size |U|+|T| x |U|+|T|
# @param LMax, maximum length for a path from a user node to another
# @param number_of_users: number of user nodes
# @param number_of_items: number of item nodes
def graph_node_similarity(adjacency_matrix: list, LMax: int, number_of_users: int, number_of_items: int) -> list:
    # Note that W is adjacency_matrix at |U|+1 to |U|+|T|
    W = parse_matrix(adjacency_matrix, 
                        0, 
                        number_of_users,
                        number_of_users,
                        number_of_users+number_of_items
                    )
    
    WT = parse_matrix(adjacency_matrix,
                        number_of_users,
                        number_of_users+number_of_items,
                        0,
                        number_of_users
                    )

    return calculate_similarity(W, WT, LMax)


def calculate_similarity(W: list, WT: list, LMax: int) -> list:
    L = 2
    M = multiply_matrix(W, WT)

    while L != LMax:
        WWT = multiply_matrix(W, WT)
        M = multiply_matrix(WWT, M)
        L = L + 2
    
    return M


# @param matrix_A = 2 dimensional matrix of size A X B
# @param matrix_B = 2 dimensional matrix of size B X C
# @return result = matrix_A x matrix_B of size A x C
def multiply_matrix(matrix_A: list, matrix_B: list, ) -> list:
    result = []

    for m in range(len(matrix_A)):
        result.append([])
        for r in range(len(matrix_B[0])):
            result[m].append(0)
    
            for k in range(len(matrix_A[m])):
                result[m][r] += matrix_A[m][k] * matrix_B[k][r]
    
    return result


# end_row and end_column excluded
def parse_matrix(matrix: list, start_row: int, end_row: int, 
    start_column: int, end_column: int) -> list:
    
    result = matrix[start_row:end_row]

    for i in range(len(result)):
        result[i] = result[i][start_column:end_column]

    return result


def generate_matrix(n: int, m: int, default: int = 0) -> list:
    matrix = []

    for i in range(n):
        temp = []
        for j in range(m):
            temp.append(default)
        matrix.append(temp)
    
    return matrix


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(f"{matrix[i][j]:.2f}", end=" ")
        print()


if __name__ == '__main__':

    start_time = time.process_time()
    CURRENT_USER = '1004'
    data, users, items, contexts, max_rating = pre_process("ratings.csv")
    current_user_index = users.index(CURRENT_USER)
    rating_matrix, items = split_items(data, users, items, contexts, max_rating)
    adjacency_matrix = construct_graph_adjacency_matrix(rating_matrix)
    similarity_scores = graph_node_similarity(adjacency_matrix, 2, len(users), len(items))
    
    selected_index = generate_recommendation(
        current_user_index, 
        similarity_scores, 
        rating_matrix,
        2,
        items,
        ("Weekday","Home","Family")
    )

    if selected_index:
        print(items[selected_index])
    else:
        print("No matching recommendation")
    end_time = time.process_time()

    print("CPU TIME: {}".format(end_time-start_time))