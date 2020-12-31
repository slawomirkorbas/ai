package com.ai.tictactoe;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Engine capable to find next best possible move with the use of Min/Max algorithm
 * This engine may be potentially used to generate games.
 */
public class MinMaxTicTacToeEngine
{
    final Integer TOTAL_FIELDS = 9;
    final Integer COMPUTER_WIN  = 1;
    final Integer COMPUTER_LOST = -1;
    final Integer DRAW = 0;
    final Integer NOT_FINISHED  = 728;
    String computerFigure;

    private class BoardPosistion
    {
        Integer row;
        Integer col;

        BoardPosistion(int r, int c)
        {
            this.row = r;
            this.col = c;
        }
    }

    /**
     * Return null if there is no move possible as the matrix is full.
     * @param board - current game state
     * @param figure - actual figure for the next move (engine)
     * @return array with coordinates for the best move [0]-row [1]-col
     */
    public Integer[] findBestMove(final String[][] board, final String figure)
    {
        computerFigure = figure;
        List<BoardPosistion> bestPositions = new ArrayList<>();
        Integer[] bestMove = null;
        Integer maxPts = null;
        if( gameState(board) == NOT_FINISHED)
        {
            for(int r = 0; r < board.length; r++)
            {
                for(int c = 0; c < board.length; c++)
                {
                    if(board[r][c].trim().isEmpty())
                    {
                        String[][] matrixCopy = copyMatrix(board);
                        matrixCopy[r][c] = figure;
                        Integer pts = evaluateGames(matrixCopy, toggle(figure), 0);
                        if(maxPts == null || pts == maxPts)
                        {
                            maxPts = pts;
                            bestPositions.add(new BoardPosistion(r,c));
                        }
                        else if(pts > maxPts)
                        {
                            maxPts = pts;
                            bestPositions.clear();
                            bestPositions.add(new BoardPosistion(r,c));
                        }
                    }
                }
            }
        }

        if(bestPositions.size() > 0)
        {
            // pick randomly best move from the list of best moves collected...
            int randomIndex = bestPositions.size() == 1 ? 0 : (new Random()).nextInt(bestPositions.size() - 1);
            BoardPosistion pos = bestPositions.get(randomIndex);
            bestMove = new Integer[2];
            bestMove[0] = pos.row;
            bestMove[1] = pos.col;
        }
        return bestMove;
    }

    /**
     * Evaluate possible games and applies score to each of them...
     * @param board - current game state
     * @param figure - actual figure
     * @param result - total score
     * @return
     */
    public Integer evaluateGames(final String[][] board, final String figure, Integer result)
    {
        Integer min = null;
        Integer max = null;
        Integer gameResult = gameState(board);
        if( gameResult == COMPUTER_LOST )
        {
            result = -10;
        }
        else if( gameResult == COMPUTER_WIN )
        {
            result = 10;
        }
        else if( gameResult == DRAW )
        {
            result = 0;
        }
        else if( gameResult == NOT_FINISHED )
        {
            for( int r=0; r < board.length; r++ )
            {
                for( int c=0; c < board.length; c++ )
                {
                    String[][] boardCopy = copyMatrix(board);
                    if( boardCopy[r][c].trim().isEmpty())
                    {
                        boardCopy[r][c] = figure;
                        int pts = evaluateGames(boardCopy, toggle(figure), result);
                        min = min == null ? pts : min;
                        max = max == null ? pts : max;
                        if( pts < min )
                        {
                            min = pts;
                        }
                        else if( pts > max )
                        {
                            max = pts;
                        }
                    }
                }
            }
            result += ( isOpponentsTurn(figure) ? min : max );
        }
        return result;
    }

    boolean isOpponentsTurn(final String currentFigure)
    {
        return  !currentFigure.equals(computerFigure);
    }

    String toggle(final String figure)
    {
        return figure.equals("x") ? "o" : "x";
    }



    /**
     * Check game state.
     * @param board - game state
     * @return 0 - game is not finished, 1 - 'x' has won, -1 - 'o' has won
     */
    Integer gameState(final String[][] board )
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
                return computerFigure.equals("x") ? COMPUTER_WIN : COMPUTER_LOST;
            if(countO == winCount )
                return computerFigure.equals("o") ? COMPUTER_WIN : COMPUTER_LOST;
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
                return computerFigure.equals("x") ? COMPUTER_WIN : COMPUTER_LOST;
            if(countO == winCount )
                return computerFigure.equals("o") ? COMPUTER_WIN : COMPUTER_LOST;
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
                    return computerFigure.equals("x") ? COMPUTER_WIN : COMPUTER_LOST;
                if(countO == winCount )
                    return computerFigure.equals("o") ? COMPUTER_WIN : COMPUTER_LOST;
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
                    return computerFigure.equals("x") ? COMPUTER_WIN : COMPUTER_LOST;
                if(countO == winCount )
                    return computerFigure.equals("o") ? COMPUTER_WIN : COMPUTER_LOST;
            }
            countX = countO = 0;
        }

        if( matrixFull(board))
        {
            return DRAW;
        }

        return NOT_FINISHED;    // game is not over or matrix is full...
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
    String[][] copyMatrix( final String[][] board )
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
