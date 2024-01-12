import pandas as pd
import numpy as np
import cv2
import os
from keras.models import load_model
model = load_model('xo_detection.h5')

df = pd.read_csv('./Submission.csv')
def is_winner(board, player):
    # Check rows, columns, and diagonals for a winner
    for i in range(3):
        if all(board[i * 3 + j] == player for j in range(3)) or all(board[j * 3 + i] == player for j in range(3)):
            return True
    if all(board[i * 3 + i] == player for i in range(3)) or all(board[i * 3 + (2 - i)] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    return all(cell != 2 for cell in board)

def minimax(board, depth, is_maximizing_player,player):
    # if is_winner(board, 0):
    #     return -1
    # if is_winner(board, 1):
    #     return 1
    if is_winner(board,player):
        return 10-depth
    if is_winner(board,not bool(player)):
        return depth-10
    if is_board_full(board):
        return 0

    if is_maximizing_player:
        max_eval = float('-inf')
        for i in range(9):
            if board[i] == 2:
                board[i] = 1
                eval = minimax(board, depth + 1, False,player)
                board[i] = 2
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(9):
            if board[i] == 2:
                board[i] = 0
                eval = minimax(board, depth + 1, True,player)
                board[i] = 2
                min_eval = min(min_eval, eval)
        return min_eval

def find_best_move_x(board):
    best_val = float('-inf')
    best_move = -1

    for i in range(9):
        if board[i] == 2:
            board[i] = 0
            if is_winner(board, 0):
                board[i] = 2
                return i
            board[i] = 2

    for i in range(9):
        if board[i] == 2:
            board[i] = 0
            for j in range(9):
                if board[j]==2:
                    board[j]=1
                    if is_winner(board,1):
                        board[j]=2
                        board[i]=2
                        return j
                    board[j]=2
            board[i] = 2

            
            eval = minimax(board, 0, True,0)
            board[i] = 2

            if eval > best_val:
                best_move = i
                best_val = eval

    return best_move
def find_best_move_o(board):
    best_val = float('-inf')
    best_move = -1

    for i in range(9):
        if board[i] == 2:
            board[i] = 1
            if is_winner(board, 1):
                board[i] = 2
                return i
            board[i] = 2

    for i in range(9):
        if board[i] == 2:
            board[i] = 1
            for j in range(9):
                if board[j]==2:
                    board[j]=0
                    if is_winner(board,0):
                        board[j]=2
                        board[i]=2
                        return j
                    board[j]=2
            board[i] = 2
            
            eval = minimax(board, 0, False,1)
            board[i] = 2

            if eval > best_val:
                best_move = i
                best_val = eval

    return best_move

def print_board(board):
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            if board[index] == 0:
                print("X", end=" | ")
            elif board[index] == 1:
                print("O", end=" | ")
            else:
                print(" ", end=" | ")
        print("\n---------")

def tell_winner(board):
    count = 0
    for i in range(9):
        if board[i] != 2:
            count += 1
    if count % 2 == 0:
        turn = 0  # 0 represents X, 1 represents O
    else:
        turn = 1  # 0 represents X, 1 represents O
    # print_board(board)
    while not is_winner(board, 0) and not is_winner(board, 1) and not is_board_full(board):
        player = turn % 2  # Alternates between 0 and 1
        # if player == 0:
        #     print("\nX's turn (Optimal Move):")
        # else:
        #     print("\nO's turn (Optimal Move):")
        move = -1
        if player==0 : move = find_best_move_x(board)
        else : move = find_best_move_o(board)
        # print(f"Optimal Move: {move + 1}")

        board[move] = player
        # print_board(board)

        turn += 1

    # Determine the winner or if it's a draw
    if is_winner(board, 0):
        return 1
    elif is_winner(board, 1):
        return 0
    else:
        return 2

for i in range(4520):
    print(str(i)+"iteration")
    image_path = "icg-freshers-data-science-competition/Dataset/Test/" + str(i) + ".png"
    if os.path.exists(image_path):
        im = cv2.imread("icg-freshers-data-science-competition/Dataset/Test/"+str(i)+".png")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        tiles = []

        tiles.append(im[59:176, 144:261].copy())
        tiles.append(im[59:176, 269:382].copy())
        tiles.append(im[59:176, 389:512].copy())
        tiles.append(im[183:296, 144:261].copy())
        tiles.append(im[183:296, 269:382].copy())
        tiles.append(im[183:296, 389:512].copy())
        tiles.append(im[303:427, 144:261].copy())
        tiles.append(im[303:427, 269:382].copy())
        tiles.append(im[303:427, 389:512].copy())

        pred=[]
        for j in tiles:
            j = cv2.resize(j, (28, 28), interpolation=cv2.INTER_AREA)  # Resize and preserve aspect ratio
            j = j.reshape((1, 28, 28))
            noise=0
            for k in range(28):
                for l in range(28):
                    noise+=abs((j[0][k][l]-30))
            
            # j = j / 255.0
            if noise<1000:
                print(noise)
                pred.append(2)
                continue
            predictions = model.predict(j/255.0)
            # print(predictions)
            predicted_label = int(np.argmax(predictions))
            pred.append(predicted_label)
        # print(pred)
        for z in range(len(pred)):
            if pred[z]==0:
                pred[z]=1
            elif pred[z]==1:
                pred[z]=0
        pred_copy = pred.copy()
        winner = tell_winner(pred_copy)
        row_values = [i] + pred + [winner]

        # print(row_values)
        # print(df.columns)
        if len(row_values) == len(df.columns):
            df.loc[len(df.index)] = row_values
        else:
            print("Length of row_values does not match the number of columns in the DataFrame.")
        # break

df.to_csv('Submission.csv',index=False)

# for i in range(4520):
#     image_path = "./icg-freshers-data-science-competition/Dataset/Test/" + str(i) + ".png"
#     if os.path.exists(image_path):
#         df['ID'][i] = i
#     else:
#         #remove row
#         df.drop(df.index[i], inplace=True)

# df.to_csv('Submission.csv',index=False)
