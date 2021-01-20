package com.ai.tictactoe.game;

/**
 * Base class for TicTacToe agents
 */
public abstract class TicTacToeAgent
{
    final Integer TOTAL_FIELDS  = 9;
    final public String playAs;

    public abstract BoardCell getNextMove(final String[][] board);

    public TicTacToeAgent(final String playAs)
    {
        this.playAs = playAs;
    }

    /**
     * Finds next possible move and updates given board.
     * @param board - initial board with game state to process
     * @return game result after the move has been made
     */
    public GameResult doMove(final String[][] board)
    {
        GameResult result = gameState(board);
        if(result == GameResult.CONTINUE)
        {
            BoardCell nextMove = this.getNextMove(board);
            if(nextMove != null)
            {
                board[nextMove.row][nextMove.col] = this.playAs;
                result = gameState(board);
            }
        }
        return result;
    }

    /**
     * Check game state.
     * @param board - game state
     * @return 0 - game is not finished, 1 - 'x' has won, -1 - 'o' has won
     */
    public GameResult gameState(final String[][] board )
    {
        // horizontal scan...
        int countX = 0, countO = 0, winCount = 3;
        for( int r=0; r < board.length; r++ )
        {
            for( int c=0; c < board.length; c++ )
            {
                countX = board[r][c].equals("x") ? countX + 1 : 0;
                countO = board[r][c].equals("o") ? countO + 1 : 0;
            }
            if(countX == winCount )
                return playAs.equals("x") ? GameResult.WIN : GameResult.LOST;
            if(countO == winCount )
                return playAs.equals("o") ? GameResult.WIN : GameResult.LOST;
            countX = countO = 0;
        }
        //vertical scan...
        countX = countO = 0;
        for(int c=0; c < board.length; c++)
        {
            for( int r=0; r < board.length; r++ )
            {
                countX = board[r][c].equals("x") ? countX + 1 : 0;
                countO = board[r][c].equals("o") ? countO + 1 : 0;
            }
            if(countX == winCount )
                return playAs.equals("x") ? GameResult.WIN : GameResult.LOST;
            if(countO == winCount )
                return playAs.equals("o") ? GameResult.WIN : GameResult.LOST;
            countX = countO = 0;
        }
        //diagonal scan left top
        countX = countO = 0;

        //scan diagonal line  (top-left)
        for( int r=0; r < board.length; r++ ) {
            int colStart = 0;
            int rowStart = r;
            for(int x=colStart, y=rowStart; x < board.length && y < board.length; x++, y++ ) {
                countX = board[x][y].equals("x") ? countX + 1 : 0;
                countO = board[x][y].equals("o") ? countO + 1 : 0;
                if(countX == winCount )
                    return playAs.equals("x") ? GameResult.WIN : GameResult.LOST;
                if(countO == winCount )
                    return playAs.equals("o") ? GameResult.WIN : GameResult.LOST;
            }
            countX = countO = 0;
        }

        //scan diagonal line  (bottom-right)
        for( int r=0; r < board.length; r++ )
        {
            int colStart = board.length - 1;
            int rowStart = r;
            for( int x=colStart, y=rowStart; x < board.length && y < board.length; x--, y++ ) {
                countX = board[x][y].equals("x") ? countX + 1 : 0;
                countO = board[x][y].equals("o") ? countO + 1 : 0;
                if(countX == winCount )
                    return playAs.equals("x") ? GameResult.WIN : GameResult.LOST;
                if(countO == winCount )
                    return playAs.equals("o") ? GameResult.WIN : GameResult.LOST;
            }
            countX = countO = 0;
        }

        if( matrixFull(board))
        {
            return GameResult.DRAW;
        }

        return GameResult.CONTINUE;    // game is not over or matrix is full...
    }

    /**
     * Returns 'true' if the matrix is full
     * @param board
     * @return
     */
    boolean matrixFull(final String[][] board)
    {
        // check if matrix is full...
        return countOccupiedFields(board) == TOTAL_FIELDS ;
    }

    /**
     * Returns number of fields occupied
     * @param board
     * @return
     */
    int countOccupiedFields(final String[][] board)
    {
        int count = 0;
        for( int r=0; r < board.length; r++ )
        {
            for( int c=0; c < board.length; c++ )
            {
                count = !board[r][c].trim().isEmpty() ? count + 1 : count;
            }
        }
        return count;
    }


    /**
     * Copies arrays - deep copy of two dimensional array
     * @param board
     * @return copy of the board
     */
    public static String[][] copyBoard( final String[][] board )
    {
        final String[][] boardCopy = new String[3][3];
        for( var r=0; r < board.length; r++ )
        {
            for( var c=0; c < board.length; c++ )
            {
                boardCopy[r][c] =  board[r][c];
            }
        }
        return boardCopy;
    }
}
