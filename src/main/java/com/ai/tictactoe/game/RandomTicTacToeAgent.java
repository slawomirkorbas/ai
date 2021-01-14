package com.ai.tictactoe.game;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Agent playing Tic-TAc-Toe by picking randomly one of the available fields.
 * May be used to evaluate performance of given neural network in terms of won games percentage.
 * Also may be used in reinforcement learning.
 *
 */
public class RandomTicTacToeAgent extends TicTacToeAgent
{
    public RandomTicTacToeAgent(final String playAs)
    {
        super(playAs);
    }

    @Override
    public BoardCell getNextMove(final String[][] board)
    {
        BoardCell freeCell = null;
        List<BoardCell> freeCells = new ArrayList<>();
        if(gameState(board) == GameResult.CONTINUE)
        {
            for(int r = 0; r < board.length; r++)
            {
                for(int c = 0; c < board.length; c++)
                {
                    if(board[r][c].trim().isEmpty())
                    {
                        freeCells.add(new BoardCell(r,c));
                    }
                }
            }
            if( freeCells.size() > 0)
            {
                freeCell = freeCells.get( freeCells.size() == 1 ? 0 : (new Random()).nextInt(freeCells.size() - 1));
            }
        }
        return freeCell;
    }
}
