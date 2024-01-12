# import pandas as pd

# df = pd.read_csv('./icg-freshers-data-science-competition/Dataset/Train/Grid_labels.csv')
# x = df.drop(['ID','Decision'],axis=1)
# y = df['Decision']

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

# Simulate the remaining game

current_board = [0, 0, 1, 0, 1, 1, 2, 0, 2]






count = 0
for i in range(9):
    if current_board[i] != 2:
        count += 1
if count % 2 == 0:
    turn = 0  # 0 represents X, 1 represents O
else:
    turn = 1  # 0 represents X, 1 represents O
print_board(current_board)
while not is_winner(current_board, 0) and not is_winner(current_board, 1) and not is_board_full(current_board):
    player = turn % 2  # Alternates between 0 and 1
    if player == 0:
        print("\nX's turn (Optimal Move):")
    else:
        print("\nO's turn (Optimal Move):")
    move = -1
    if player==0 : move = find_best_move_x(current_board)
    else : move = find_best_move_o(current_board)
    print(f"Optimal Move: {move + 1}")

    current_board[move] = player
    print_board(current_board)

    turn += 1

# Determine the winner or if it's a draw
if is_winner(current_board, 0):
    print("\nX is the winner! : 1")
elif is_winner(current_board, 1):
    print("\nO is the winner! : 0")
else:
    print("\nIt's a draw! : 2")

#0 => o win
#1 => x win
#2 => draw