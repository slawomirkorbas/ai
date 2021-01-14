package com.ai.tictactoe.game;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Engine capable to find next best possible move with the use of Min/Max algorithm
 * This engine may be potentially used to generate games.
 */
public class MinMaxTicTacToeAgent extends TicTacToeAgent
{
    public MinMaxTicTacToeAgent(final String playAs)
    {
        super(playAs);
    }

    /**
     * Return null if there is no move possible as the matrix is full. Otherwise returns one (random) of the
     * best scored moves from computed list.
     * @param board - current game state
     * @return BoardCell
     */
    @Override
    public BoardCell getNextMove(final String[][] board)
    {
        List<BoardCell> bestPositions = computeBestMoves(board, this.playAs);
        if(bestPositions.size() > 0)
        {
            // pick randomly move from the list of best moves collected...
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
        List<BoardCell> bestPositions = new ArrayList<>();
        Integer maxPts = null;
        if(gameState(board) == GameResult.CONTINUE)
        {
            for(int r = 0; r < board.length; r++)
            {
                for(int c = 0; c < board.length; c++)
                {
                    if(board[r][c].trim().isEmpty())
                    {
                        String[][] matrixCopy = copyBoard(board);
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
        GameResult gameResult = gameState(board);
        if( gameResult == GameResult.LOST )
        {
            result = -10;
        }
        else if( gameResult == GameResult.WIN )
        {
            result = 10;
        }
        else if( gameResult == GameResult.DRAW )
        {
            result = 0;
        }
        else if( gameResult == GameResult.CONTINUE )
        {
            for( int r=0; r < board.length; r++ )
            {
                for( int c=0; c < board.length; c++ )
                {
                    String[][] boardCopy = copyBoard(board);
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
        return  !currentFigure.equals(playAs);
    }

    String toggle(final String figure)
    {
        return figure.equals("x") ? "o" : "x";
    }
}
