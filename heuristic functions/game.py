import copy
import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    pieces = ['b', 'r']
    max_depth = 2

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.board = [[' ' for j in range(5)] for i in range(5)]
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ 
        Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        move = []

        best_move = self.max_value(state, 0)

        newState = best_move[1][0]
        newRow = best_move[1][1]
        newCol = best_move[1][2]
        oldRow = best_move[1][3]
        oldCol = best_move[1][4]

        if oldRow == -1:
            move = [(newRow, newCol)]
        else:
            move = [(newRow, newCol), (oldRow, oldCol)]

        return move

    def succ(self, state, my_piece): 
        """
        Generates a list of valid successors for the current game state 
        on placing your piece. (defined by self.my_piece)
        """
        #tuples of successors to return in the form (state, new row, new column, old row, old column)
        successors = []

        #Count the red pieces to determine phase
        phase = 'drop'
        red_count = 0

        for i, row in enumerate(state):
            for j in range(len(row)):
                if state[i][j] == my_piece:
                    red_count += 1

        if red_count == 4:
            phase = 'continued'

        #if it's drop phase, generate a game state for each open space
        if phase =='drop':
            for i, row in enumerate(state):
                for j in range(len(row)):
                    if state[i][j] == ' ':
                        newState = copy.deepcopy(state)
                        newState[i][j] = my_piece
                        successors.append((newState, i, j, -1, -1))

        #if it's the continuous phase, generate a game state for each piece's possible moves
        else:
            for i,row in enumerate(state):
                for j in range(len(row)):
                    if state[i][j] == my_piece:

                        #generate new state for each empty space to left of current piece
                        if i > 0:
                            if state[i - 1][j] == ' ':
                                newState = copy.deepcopy(state)
                                newState[i - 1][j] = my_piece
                                newState[i][j] = ' '
                                successors.append((newState, i - 1, j, i, j))

                            if j > 0:
                                if state[i - 1][j - 1] == ' ':
                                    newState = copy.deepcopy(state)
                                    newState[i - 1][j - 1] = my_piece
                                    newState[i][j] = ' '
                                    successors.append((newState, i - 1, j - 1, i, j))
                            
                            if j < 4:
                                if state[i - 1][j + 1] == ' ':
                                    newState = copy.deepcopy(state)
                                    newState[i - 1][j + 1] = my_piece
                                    newState[i][j] = ' '
                                    successors.append((newState, i - 1, j + 1, i, j))

                        #generate new state for each empty space to right of current piece
                        if i < 4:
                            if state[i + 1][j] == ' ':
                                newState = copy.deepcopy(state)
                                newState[i + 1][j] = my_piece
                                newState[i][j] = ' '
                                successors.append((newState, i + 1, j, i, j))

                            if j > 0:
                                if state[i + 1][j - 1] == ' ':
                                    newState = copy.deepcopy(state)
                                    newState[i + 1][j - 1] = my_piece
                                    newState[i][j] = ' '
                                    successors.append((newState, i + 1, j - 1, i, j))

                            if j < 4:
                                if state[i + 1][j + 1] == ' ':
                                    newState = copy.deepcopy(state)
                                    newState[i + 1][j + 1] = my_piece
                                    newState[i][j] = ' '
                                    successors.append((newState, i + 1, j + 1, i, j))

                        #generate new state for empty spaces above and below the current piece
                        if j > 0:
                            if state[i][j - 1] == ' ':
                                    newState = copy.deepcopy(state)
                                    newState[i][j - 1] = my_piece
                                    newState[i][j] = ' '
                                    successors.append((newState, i, j - 1, i, j))
                        
                        if j < 4:
                            if state[i][j + 1] == ' ':
                                    newState = copy.deepcopy(state)
                                    newState[i][j + 1] = my_piece
                                    newState[i][j] = ' '
                                    successors.append((newState, i, j + 1, i, j))

        return successors
    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    
    def heuristic_game_value(self, state):
        """ 
        Define the heuristic game value of the current board state taking into account players
        and opponents

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            float heuristic_val (heuristic computed for the game state)
        """

        heuristic_val = 0

        #Heuristic:
        #Line: single piece = 1, subsequent pieces in a row *= 4
        #Square: 
        #subtract oppenent's score from heuristic

        square_score_black = 0
        square_score_red = 0
        for i,row in enumerate(state):
            for j in range(len(row)):
                #SQUARE scoring: for each piece, add 1 to its score for each same player's piece within 1 tile
                if state[i][j] == 'b':
                    square_score_black = 1
                    #only has pieces to the left if its not in left column
                    if i > 0:
                        if state[i - 1][j] == 'b':
                            square_score_black += 1
                        
                        if j > 0:
                            if state[i - 1][j - 1] == 'b':
                                square_score_black += 1
                        
                        if j < 4:
                            if state[i - 1][j + 1] == 'b':
                                square_score_black += 1
                    
                    #now same for right side
                    if i < 4:
                        if state[i + 1][j] == 'b':
                            square_score_black += 1

                        if j > 0:
                            if state[i + 1][j - 1] == 'b':
                                square_score_black += 1

                        if j < 4:
                            if state[i + 1][j + 1] == 'b':
                                square_score_black += 1

                    #now same for top and bottom
                    if j > 0:
                        if state[i][j - 1] == 'b':
                            square_score_black += 1
                    
                    if j < 4:
                        if state[i][j + 1] == 'b':
                            square_score_black += 1

                #Scoring for red
                #SQUARE scoring: for each piece, add 1 to its score for each same player's piece within 1 tile
                elif state[i][j] == 'r':
                    square_score_red = 1
                    #only has pieces to the left if its not in left column
                    if i > 0:
                        if state[i - 1][j] == 'r':
                            square_score_red += 1
                        
                        if j > 0: 
                            if state[i - 1][j - 1] == 'r':
                                square_score_red += 1
                        
                        if j < 4: 
                            if state[i - 1][j + 1] == 'r':
                                square_score_red += 1
                    
                    #now same for right side
                    if i < 4:
                        if state[i + 1][j] == 'r':
                            square_score_red += 1

                        if j > 0:
                            if state[i + 1][j - 1] == 'r':
                                square_score_red += 1

                        if j < 4: 
                            if state[i + 1][j + 1] == 'r':
                                square_score_red += 1

                    #now same for top and bottom
                    if j > 0:
                        if state[i][j - 1] == 'r':
                            square_score_red += 1
                    
                    if j < 4:
                        if state[i][j + 1] == 'r':
                            square_score_red += 1

        #add scores based on self.mypiece
        if self.my_piece == 'r':
            heuristic_val += square_score_red
            heuristic_val -= square_score_black
        else:
            heuristic_val += square_score_black
            heuristic_val -= square_score_red

        #divide heuristic by max heuristic value to get floating point
        heuristic_val = heuristic_val / 16

        return heuristic_val
 
    def game_value(self, state):
        """ 
        Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        #check for 4 in a row, first piece encountered will always be top piece (end piece)
        #check black first
        i_black = -1
        j_black = -1
        black_win = False
        for i, row in enumerate(state):
            for j in range(len(row)):
                if state[i][j] == 'b':
                    i_black = i
                    j_black = j
                    break
            if i_black != -1:
                break

        #if no black has been placed, there is no winner yet (still in drop phase)
        if i_black == -1:
            return 0
        
        #check diagonal towards bottom left
        if i_black < 2:
            if j_black > 2:
                if state[i_black + 1][j_black - 1] == state[i_black + 2][j_black - 2] == state[i_black + 3][j_black - 3] == 'b':
                    black_win = True

            #check straight down
            if state[i_black + 1][j_black] == state[i_black + 2][j_black] == state[i_black + 3][j_black] == 'b':
                black_win = True
            
            #check down and to the right
            if j_black < 2:
                if state[i_black + 1][j_black + 1] == state[i_black + 2][j_black + 2] == state[i_black + 3][j_black + 3] == 'b':
                    black_win = True

        #check straight to the right
        if j_black < 2:
            if state[i_black][j_black + 1] == state[i_black][j_black + 2] == state[i_black][j_black + 3] == 'b':
                black_win = True

        #for square: first piece encountered will always be top left piece
        if i_black < 4:
            if j_black < 4:
                if state[i_black][j_black + 1] == state[i_black + 1][j_black] == state[i_black + 1][j_black + 1] == 'b':
                    black_win = True

        #check red next
        i_red = -1
        j_red = -1
        red_win = False
        for i, row in enumerate(state):
            for j in range(len(row)):
                if state[i][j] == 'r':
                    i_red = i
                    j_red = j
                    break
            if i_red != -1:
                break

        #if no red has been placed, there is no winner yet (still in drop phase)
        if i_red == -1:
            return 0
        
        #check diagonal towards bottom left
        if i_red < 2:
            if j_red > 2:
                if state[i_red + 1][j_red - 1] == state[i_red + 2][j_red - 2] == state[i_red + 3][j_red - 3] == 'r':
                    red_win = True

            #check straight down
            if state[i_red + 1][j_red] == state[i_red + 2][j_red] == state[i_red + 3][j_red] == 'r':
                red_win = True
            
            #check down and to the right
            if j_red < 2:
                if state[i_red + 1][j_red + 1] == state[i_red + 2][j_red + 2] == state[i_red + 3][j_red + 3] == 'r':
                    red_win = True

        #check straight to the right
        if j_red < 2:
            if state[i_red][j_red + 1] == state[i_red][j_red + 2] == state[i_red][j_red + 3] == 'r':
                red_win = True

        #for square: first piece encountered will always be top left piece
        if i_red < 4:
            if j_red < 4:
                if state[i_red][j_red + 1] == state[i_red + 1][j_red] == state[i_red + 1][j_red + 1] == 'r':
                    red_win = True

        #return -1, 0, or 1 based on winner and the colour that we are playing
        if self.my_piece == 'b':
            if black_win:
                return 1
            elif red_win:
                return -1
        elif self.my_piece == 'r':
            if black_win:
                return -1
            elif red_win:
                return 1
            
        return 0 # no winner yet
    
    def max_value(self, state, depth):
        """
        The helper function to implement min-max
        """
        #check if game is already over at this depth
        game_value = self.game_value(state)
        if game_value != 0:
            return (game_value, None)

        current_depth = depth
        turn_colour = ''
        my_turn = False

        #determine the colour of the pieces that play in this turn
        if current_depth % 2 == 0:
            turn_colour = self.my_piece
            my_turn = True
        elif self.my_piece == 'r':
            turn_colour = 'b'
        else:
            turn_colour = 'r'

        successors = self.succ(state, turn_colour)

        #if we're at the max depth, return own heuristic
        if current_depth == self.max_depth:
            return (self.heuristic_game_value(state), None)

        #if not at max depth, call max_value on all successors
        if current_depth < self.max_depth:
            heuristics = []
            for i, state in enumerate(successors):
                heuristics.append(self.max_value(successors[i][0], current_depth + 1)[0])
                
            max_heuristic = max(heuristics)
            min_heuristic = min(heuristics)

            max_index = heuristics.index(max_heuristic)
            min_index = heuristics.index(min_heuristic)

            #return the index of the successor with the highest or lowest heuristic, depending on turn
            if my_turn:
                return (max_heuristic, successors[max_index])
            else:
                return (min_heuristic, successors[min_index])
        



############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
