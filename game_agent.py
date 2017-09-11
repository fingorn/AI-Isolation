"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
#    print("Custom score used")
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """






    if game.is_loser(player):
        return float('-inf')
    if game.is_winner(player):
        return float('inf')
    
    legal_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    
    num_lm = len(legal_moves)
    num_om = len(opponent_moves)
    
    #Overlapping moves
    overlaps = [ g for g in legal_moves if g in opponent_moves ]
    om = 1
    if overlaps:
        om *= len(overlaps) 
        #if there is only 1 move left for the opponent, and our move overlaps with it, it is a winning move
        if num_om == 1:
            return float('inf')
        
    
    #Increase the score by the number of overlaps
    return  om + float(num_lm) -  2 * float(num_om) 

def custom_score_2(game, player):
    #    print("Custom score 2 used")
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_loser(player):
        return float('-inf')
    if game.is_winner(player):
        return float('inf')
    
    legal_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    
    
    num_om = len(opponent_moves)
    
    
    total_sq_dist = 0
    
    for row,c in opponent_moves:
        dist_from_center = ( row - (game.height -1) /2  , c - (game.width-1)/2 )
        sq_dist = pow(dist_from_center[0],2) + pow(dist_from_center[1],2)
        total_sq_dist += sq_dist
        
    avg_opp_sq_dist_from_center = int(total_sq_dist / num_om) if num_om != 0 else 0    
        
    
    
    return float(len(legal_moves)) + avg_opp_sq_dist_from_center - 2 * float(num_om)




def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    
    if game.is_loser(player):
        return float('-inf')
    if game.is_winner(player):
        return float('inf')
    
    
    legal_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    num_om = len(opponent_moves)
    #Produce a score that is higher when the oponent moves are closer to the edge, 
    #in other words, if more opponent moves are further from the center (3,3) , produce a higher score
    
    total_sq_dist = 0
    
    for row,c in opponent_moves:
        dist_from_center = ( row - (game.height -1) /2  , c - (game.width-1)/2 )
        sq_dist = pow(dist_from_center[0],2) + pow(dist_from_center[1],2)
        total_sq_dist += sq_dist
    avg_opp_sq_dist_from_center = int(total_sq_dist / num_om) if num_om != 0 else 0
    #Overlapping moves
    overlaps = [ g for g in legal_moves if g in opponent_moves ]
    #Chase opponent by choosing moves that overlaps with the opponent's move
    om = 1
    if overlaps: 
        om = len(overlaps)
        if num_om == 1:
            #If opponent have only 1 move left and that overlaps with your move, that's your winning move
            return float('inf')
    return om  +  float(len(legal_moves)) + avg_opp_sq_dist_from_center - 2*float(num_om)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    
#    def score(self,game,player):
#            
#        return custom_score_3(game,player)
    
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        bestMove = (-1,-1)
        bestValue = float('-inf')
        
        forecast_moves = game.get_legal_moves(self)
        if not forecast_moves:
            return bestMove
        
        for forecast_move in forecast_moves:
    
            value = self.min_value(game.forecast_move(forecast_move),depth-1)
            if value > bestValue:
                bestValue = value
                bestMove = forecast_move
                
        return bestMove
        
#        return random.choice(game.get_legal_moves())
    

#Test 1
    
    def max_value(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        #terminal check
        if depth==0 or game.utility(game.inactive_player) != 0.:
            #Reached terminal state
            return self.score(game,self)
            
        
        bestValue = float('-inf')
        
        for forecast_move in game.get_legal_moves(self):
            value = self.min_value(game.forecast_move(forecast_move), depth-1)
            if value > bestValue:
                bestValue = value
        return bestValue
        
    
    def min_value(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #terminal check
        if depth==0 or game.utility(game.inactive_player) != 0.:
            #Reached terminal state
            return self.score(game,self)
         
        minValue = float('inf')
        
        for forecast_move in game.get_legal_moves(game.get_opponent(self)):
            value = self.max_value(game.forecast_move(forecast_move),depth-1)
            if value < minValue:
                minValue = value
                
        return minValue
    
      


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        
        
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            #Implement iterative deepening
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            # Handle any actions required after timeout as needed
            return best_move

        # Return the best move from the last completed search iteration
        return best_move
    
    
    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
                
        bestMove = (-1,-1)
        bestValue = float('-inf')
        
        forecast_moves = game.get_legal_moves(self)
        if not forecast_moves:
            return bestMove
        
        for forecast_move in forecast_moves:
    
            value = self.min_value(game.forecast_move(forecast_move,float('-inf'),float('inf')),depth-1)
            if value > bestValue:
                bestValue = value
                bestMove = forecast_move
                
        return bestMove


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #define initial bestMove
        bestMove = (-1,-1)
        
        
        my_forecast_moves = game.get_legal_moves(self)
        if not my_forecast_moves:
            return bestMove
        
        #There are moves left for me
        
        #Can't do pruning at the root of the tree
        
        for move in my_forecast_moves:
            value = self.min_value(game.forecast_move(move), depth -1,alpha,beta)
            
            
            if value > alpha:
                alpha = value
                bestMove = move
#                
#            if value > bestScore:
#                bestScore = value
#                bestMove = move
#        
        
        return bestMove
            
        
    def min_value(self,game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0 or game.utility(game.active_player) != 0.:
            return self.score(game,self)
        
        
        minValue = float('inf')
        for opponent_move in game.get_legal_moves(game.get_opponent(self)):
            value = self.max_value(game.forecast_move(opponent_move), depth-1, alpha,beta)
                        # Beta = Minimum upper bound, can only go lower    
            if value < minValue:
                minValue = value
                
            if alpha >= value:
                #Prune tree by returning immediately
                return value
                
            #Set beta to the new lowest state and pass beta value to next sibling in the loop
            beta = min(beta, value)

                
        return minValue
    
    def max_value(self,game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0 or game.utility(game.active_player) != 0.:
            return self.score(game,self)
        
        maxValue = float('-inf')
        for my_move in game.get_legal_moves(self):
            value = self.min_value(game.forecast_move(my_move), depth-1, alpha, beta)
            if value > maxValue:
                maxValue = value
                
            if value >= beta:
                #Prune 
                return value
                        
            alpha = max(alpha, value)
            
        return maxValue
        
        
        
