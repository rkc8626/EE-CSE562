import time
import random
import io

class key:
    def key(self):
        return "10jifn2eonvgp1o2ornfdlf-1230"

class ai:
    def __init__(self):
        self.depth = 10

    class state:
        def __init__(self, a, b, a_fin, b_fin):
            self.a = a        # Player A's side of the board
            self.b = b        # Player B's side of the board
            self.a_fin = a_fin  # Player A's final score area
            self.b_fin = b_fin  # Player B's final score area

    # Method to perform a move in the game
    def move(self, a, b, a_fin, b_fin, t):
        # Open a file to append the time taken for each depth
        with open('time.txt', 'a') as f:
            best_move = None
            best_val = float('-inf')
            # Looping through different depths to analyze the game
            for d in range(1, self.depth + 1):
                f.write('depth = ' + str(d) + '\n')
                curr_state = self.state(a, b, a_fin, b_fin)
                t_start = time.time()
                # Perform the minimax search to find the best move and its value
                val, move = self.minmax(curr_state, float('-inf'), float('inf'), 0)
                # Update the best move if a better one is found
                if val > best_val:
                    best_val = val
                    best_move = move
                f.write(str(time.time() - t_start) + '\n')
            f.write('\n')
        return best_move

    # A heuristic function to evaluate the game state
    def heuristic(self, state):
        # The heuristic is the difference in scores, favoring larger differences.
        return abs(state.a_fin - state.b_fin)

    # def heuristic(self, state):
    #     stone_count_weight = 1.0  # Weight for the difference in stone count
    #     store_count_weight = 1.5  # Weight for the difference in store count
    #     empty_pits_weight = 0.8  # Weight for the number of empty pits
    #     stone_count_diff = sum(state.a) - sum(state.b)
    #     store_count_diff = state.a_fin - state.b_fin
    #     empty_pits_a = len([p for p in state.a if p == 0])
    #     empty_pits_b = len([p for p in state.b if p == 0])
    #     heuristic_value = (
    #         stone_count_weight * stone_count_diff +
    #         store_count_weight * store_count_diff +
    #         empty_pits_weight * (empty_pits_a - empty_pits_b)
    #     )
    #     return heuristic_value

    # Method to calculate the state resulting from a particular move
    def updateLocalState(self, state, move):
        # If the selected pit is empty, no move can be made.
        if state.a[move] == 0:
            return

        # Save the current board state.
        original_state = state.a[:]

        # Combine the two sides of the board including the stores to form a linear board.
        board = state.a[move:] + [state.a_fin] + state.b + state.a[:move]
        count = state.a[move]
        board[0] = 0  # Empty the pit from which the stones are picked.
        position = 1  # Start distributing from the next pit.

        # Distribute the stones in the picked pit.
        while count > 0:
            board[position] += 1
            position = (position + 1) % 13  # Wrap around the board (assuming 6 pits per side and 1 store).
            count -= 1

        # Update the final stores and the two sides of the board.
        a_fin = board[6 - move]
        b_fin = state.b_fin
        a = board[13 - move:] + board[:6 - move]
        b = board[7 - move:13 - move]

        # Flags to check if the player gets another turn or captures stones.
        gets_another_turn = False
        capture = False

        position = (position - 1) % 13
        # Check if the last stone landed in the player's store, granting another turn.
        if position == 6 - move:
            gets_another_turn = True

        # Check if the last stone results in a capture.
        if (0 <= position <= 5 - move) and original_state[move] < 14:
            index = position + move
            if (original_state[index] == 0 or position % 13 == 0) and b[5 - index] > 0:
                capture = True
        elif position >= 14 - move and original_state[move] < 14:
            index = position + move - 13
            if (original_state[index] == 0 or position % 13 == 0) and b[5 - index] > 0:
                capture = True

        # Perform capture if applicable.
        if capture:
            a_fin += a[index] + b[5 - index]
            b[5 - index] = 0
            a[index] = 0

        # Check if one side is empty and collect remaining stones to the respective store.
        if sum(a) == 0:
            b_fin += sum(b)
        elif sum(b) == 0:
            a_fin += sum(a)

        # Return the updated state (the constructor or a method should be defined elsewhere).
        return self.state(a, b, a_fin, b_fin)

    def minmax(self, state, alpha, beta, depth, maxx=True):
        # Determine if the current state is a terminal state or if the maximum depth has been reached.
        terminal_condition = state.a_fin > 36 or state.b_fin > 36 or (state.a_fin == 36 and state.b_fin == 36)
        if depth == self.depth or terminal_condition:
            # If so, return the heuristic value of the state.
            return self.heuristic(state)

        if depth == 0:
            # If we're at the root of the decision tree, initialize the best value and move.
            best_val = float('-inf') if maxx else float('inf')
            best_move = -1
            # Iterate through all possible moves (0 to 5).
            for move in range(6):
                # Get the state resulting from making a move.
                next_state = self.updateLocalState(state, move)
                if next_state:
                    # Recursively call minmax on the resulting state, swapping the maximizing player.
                    val = self.minmax(next_state, alpha, beta, depth + 1, not maxx)
                    if maxx and val > best_val:
                        # Update the best value and move for a maximizing player.
                        best_val = val
                        best_move = move
                    elif not maxx and val < best_val:
                        # Update the best value and move for a minimizing player.
                        best_val = val
                        best_move = move
                    if maxx:
                        # Update the alpha value for alpha-beta pruning.
                        alpha = max(alpha, val)
                        if alpha >= beta:
                            # Break out of the loop if alpha meets or exceeds beta.
                            break
                    else:
                        # Update the beta value for alpha-beta pruning.
                        beta = min(beta, val)
                        if beta <= alpha:
                            # Break out of the loop if beta falls below or meets alpha.
                            break
            # Return the best value and the best move after exploring all possibilities.
            return best_val, best_move
        else:
            # If we're not at the root, initialize the value for this node.
            val = float('-inf') if maxx else float('inf')
            # Iterate through all possible moves (0 to 5).
            for move in range(6):
                # Get the state resulting from making a move.
                next_state = self.updateLocalState(state, move)
                if next_state:
                    # Recursively call minmax on the resulting state, swapping the maximizing player.
                    new_val = self.minmax(next_state, alpha, beta, depth + 1, not maxx)
                    if maxx:
                        # Update the value for a maximizing player.
                        val = max(val, new_val)
                        # Update the alpha value for alpha-beta pruning.
                        alpha = max(alpha, val)
                        if alpha >= beta:
                            # Break out of the loop if alpha meets or exceeds beta.
                            break
                    else:
                        # Update the value for a minimizing player.
                        val = min(val, new_val)
                        # Update the beta value for alpha-beta pruning.
                        beta = min(beta, val)
                        if beta <= alpha:
                            # Break out of the loop if beta falls below or meets alpha.
                            break
            # Return the value for this node after exploring all possibilities.
            return val
