from enum import Enum

class Move:
    # Step 1, creating Move class
    INVALID_COORDINATE = -1

    def __init__(self, value, row=INVALID_COORDINATE, col=INVALID_COORDINATE):
        self.row = row
        self.col = col
        self.value = value
    # string representation of move, for better visualzation
    def __str__(self):
        return f"Move(row={self.row}, col={self.col}, value={self.value})"

class Player(Enum):
    # Step 2, create enumerated type
    X = 'X'
    O = 'O'

class GameState:
    # Step 3, create the game state
    def __init__(self):
        # a. Initialize the board of TicTacToe with full of None
        self.board = [[None for _ in range(3)] for _ in range(3)]
    
    def game_over(self):
        # b. method game_over, return True if gameover, else False
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            if self.board[i][0] is not None and all(self.board[i][j] == self.board[i][0] for j in range(3)):
                return True
            if self.board[0][i] is not None and all(self.board[j][i] == self.board[0][i] for j in range(3)):
                return True
        
        if self.board[0][0] is not None and all(self.board[i][i] == self.board[0][0] for i in range(3)):
            return True
        if self.board[0][2] is not None and all(self.board[i][2 - i] == self.board[0][2] for i in range(3)):
            return True

        # Check for a full board
        if all(self.board[i][j] is not None for i in range(3) for j in range(3)):
            return True

        return False
    
    def winner(self):
        # c. method winner, determinds which player wins
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            if self.board[i][0] is not None and all(self.board[i][j] == self.board[i][0] for j in range(3)):
                return self.board[i][0]
            if self.board[0][i] is not None and all(self.board[j][i] == self.board[0][i] for j in range(3)):
                return self.board[0][i]

        if self.board[0][0] is not None and all(self.board[i][i] == self.board[0][0] for i in range(3)):
            return self.board[0][0]
        if self.board[0][2] is not None and all(self.board[i][2 - i] == self.board[0][2] for i in range(3)):
            return self.board[0][2]

        return None
    
    def __str__(self):
        # d. str method to return a readable representation of the board state
        board_str = ""
        for row in self.board:
            row_str = " | ".join([player.value if player else " " for player in row])
            board_str += row_str + "\n" + "-"*9 + "\n"
        return board_str

    def spot(self, row, col):
        # e. returns the piece that is in the given position on the board
        return self.board[row][col]

    def move(self, row, col, player):
        # f. returns a new gamestate, a copy of the current one
        if self.board[row][col] is not None:
            # return none if the spot is taken
            return None
        
        new_state = GameState()
        new_state.board = [row[:] for row in self.board]
        new_state.board[row][col] = player
        return new_state
    

class TicTacToeSolver:
    # Step 4, run minimax algorithm
    def find_best_move(self, state: GameState, player: Player):
        # a. takes gamestate and a player, find out the best move at the current state
        best_move = self.solve_my_move(state, float('-inf'), float('inf'), player)
        return best_move
    
    def solve_my_move(self, state: GameState, alpha: float, beta: float, player: Player):
        # b. solving steps, following the pseudocode
        if state.game_over():
            # three conditions
            winner = state.winner()
            if winner == player:
                return Move(value=1)
            elif winner is None:
                return Move(value=0)
            else:
                return Move(value=-1)
        # keeping track of the moves initialize as None
        best_move = None

        for row in range(3):
            for col in range(3):
                if state.spot(row, col) is None:
                    # happens when a space is empty
                    new_state = state.move(row, col, player)
                    child_move = self.solve_opponent_move(new_state, alpha, beta, player)

                    if best_move is None or child_move.value > best_move.value:
                        # first move
                        best_move = Move(value=child_move.value, row=row, col=col)
                    if best_move.value > beta:
                        # determinding best move
                        return best_move
                    
                    alpha = max(alpha, best_move.value)
        
        return best_move
    
    def solve_opponent_move(self, state: GameState, alpha: float, beta: float, player: Player):
        # c. solving move for the opponent, follows closely with solve_my_move
        opponent = Player.X if player == Player.O else Player.O
        
        if state.game_over():
            winner = state.winner()
            if winner == player:
                return Move(value=1)
            elif winner is None:
                return Move(value=0)
            else:
                return Move(value=-1)
        
        best_move = None

        for row in range(3):
            for col in range(3):
                if state.spot(row, col) is None:
                    new_state = state.move(row, col, opponent)
                    child_move = self.solve_my_move(new_state, alpha, beta, player)

                    if best_move is None or child_move.value < best_move.value:
                        best_move = Move(value=child_move.value, row=row, col=col)

                    if best_move.value < alpha:
                        return best_move
                    
                    beta = min(beta, best_move.value)
        
        return best_move

# Step 5, testing
if __name__ == "__main__":
    state = GameState()
    solver = TicTacToeSolver()
    current_player = Player.X

    while not state.game_over():
        print(state)
        best_move = solver.find_best_move(state, current_player)
        if best_move.row == Move.INVALID_COORDINATE:
            print("No valid moves left.")
            break
        state = state.move(best_move.row, best_move.col, current_player)
        current_player = Player.O if current_player == Player.X else Player.X
    
    print(state)
    winner = state.winner()
    if winner:
        print(f"The winner is {winner.value}!")
    else:
        print("The game is a draw!")
