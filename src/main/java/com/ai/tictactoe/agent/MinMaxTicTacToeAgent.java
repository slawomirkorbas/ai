package com.ai.tictactoe.agent;

import com.ai.tictactoe.util.BoardCell;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Engine capable to find next best possible move with the use of Min/Max algorithm
 * This engine may be potentially used to generate games.
 */
public class MinMaxTicTacToeAgent extends TicTacToeAgent
{
    /**
     * Return null if there is no move possible as the matrix is full. Otherwise returns one (random) of the
     * best scored moves from computed list.
     * @param board - current game state
     * @param figure - actual figure for the next move (engine)
     * @return BoardCell
     */
    public BoardCell getNextMove(final String[][] board, final String figure)
    {
        List<BoardCell> bestPositions = computeBestMoves(board, figure);
        if(bestPositions.size() > 0)
        {
            // pick randomly best move from the list of best moves collected...
            int randomIndex = bestPositions.size() == 1 ? 0 : (new Random()).nextInt(bestPositions.size() - 1);
            return bestPositions.get(randomIndex);
        }
        return null;
    }

    /**
     * Return null if there is no move possible as the matrix is full. Otherwise returns first best scored
     * move from computed list.
     * @param board - current game state
     * @param figure - actual figure for the next move (engine)
     * @return BoardCell
     */
    public BoardCell getFirstNextMove(final String[][] board, final String figure)
    {
        List<BoardCell> bestPositions = computeBestMoves(board, figure);
        if(bestPositions.size() > 0)
        {
            return bestPositions.get(0);
        }
        return null;
    }

    /**
     * Compute best moves for given boar situation
     * @param board
     * @param figure
     * @return
     */
    public List<BoardCell> computeBestMoves(final String[][] board, final String figure)
    {
        computerFigure = figure;
        List<BoardCell> bestPositions = new ArrayList<>();
        Integer maxPts = null;
        if(gameState(board) == NOT_FINISHED)
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
                            bestPositions.add(new BoardCell(r,c));
                        }
                        else if(pts > maxPts)
                        {
                            maxPts = pts;
                            bestPositions.clear();
                            bestPositions.add(new BoardCell(r,c));
                        }
                    }
                }
            }
        }
        return bestPositions;
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
