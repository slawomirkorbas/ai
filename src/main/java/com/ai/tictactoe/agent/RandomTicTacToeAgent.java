package com.ai.tictactoe.agent;

import com.ai.tictactoe.util.BoardCell;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Agent playing Tic-TAc-Toe by picking randomly one of the available fields.
 * May be used to evaluate performance of given neural network in terms of won games percentage.
 * Also may be used in reinforcement learning.
 *
 */
public class RandomTicTacToeAgent extends TicTacToeAgent
{
    public BoardCell getNextMove(final String[][] board, final String figure)
    {
        computerFigure = figure;
        BoardCell freeCell = null;
        List<BoardCell> availableCells = new ArrayList<>();
        if(gameState(board) == NOT_FINISHED)
        {
            for(int r = 0; r < board.length; r++)
            {
                for(int c = 0; c < board.length; c++)
                {
                    if(board[r][c].trim().isEmpty())
                    {
                        availableCells.add(new BoardCell(r,c));
                    }
                }
            }
        }

        if(availableCells.size() > 0)
        {
            freeCell = availableCells.get((new Random()).nextInt(availableCells.size() - 1));
        }

        return freeCell;
    }

    /**
     * Generates tic-tac-toe game states (boards) for the given number of moves. The assumption is that each generated board
     * corresponds to situation where the "X" player is about to make next move and "X" player begins a game.
     * @param numberOfMoves - number of moves done
     * @param maxBatchSize - maximum number of generated boards
     *
     * @return collection of unique game states with given number of moves
     */
    public List<String[][]> generateGameStates(int numberOfMoves, int maxBatchSize)
    {
        final List <String[][]> gameStates = new ArrayList<>();
        final Map<String, String[][]> uniqueGameStateMap = new HashMap<>();
        if(numberOfMoves == 0)
        {
            gameStates.add( new String[][] {{" ", " ", " "}, {" ", " ", " "}, {" ", " ", " "}});
            return gameStates;
        }
        else
        {
            while(gameStates.size() < maxBatchSize)
            {
                String figure = "o";
                String[][] board = new String[][] {{" ", " ", " "}, {" ", " ", " "}, {" ", " ", " "}};
                for(int i = 0; i < numberOfMoves && i < 8; i++)
                {
                    BoardCell freeCell = getNextMove(board, figure);
                    if(freeCell == null)
                    {
                        break;
                    }
                    board[freeCell.row][freeCell.col] = figure;
                    figure = figure.equals("x") ? "o" : "x";
                }
                uniqueGameStateMap.put(getBoardKey(board), board);
            }
        }

        return new ArrayList<>(uniqueGameStateMap.values());
    }

    /**
     * Utility method generating unique board key.
     * @param board
     * @return
     */
    private String getBoardKey(final String[][] board)
    {
        String key = "";
        for(int r = 0; r < board.length; r++)
        {
            for(int c = 0; c < board.length; c++)
            {
                key += board[r][c];
            }
        }
        return key;
    }
}
