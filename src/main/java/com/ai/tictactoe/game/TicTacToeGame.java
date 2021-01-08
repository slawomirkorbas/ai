package com.ai.tictactoe.game;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Class storing whole ga,e execution along with the result
 */
public class TicTacToeGame implements Serializable
{
    /** current board **/
    public String[][] board =  new String[][] {{" ", " ", " "}, {" ", " ", " "}, {" ", " ", " "}};

    /** "x", "o" or null(in case of DRAW) */
    public String whoWon = null;

    /** all game states (boards) **/
    public List<String[][]> states = new ArrayList<>();

    public boolean finished = false;

    public TicTacToeGame()
    {
        // add initial game state
        states.add(TicTacToeAgent.copyBoard(this.board));
    }

    public void update(final GameResult result)
    {
        states.add(TicTacToeAgent.copyBoard(board));
        if(result != GameResult.CONTINUE)
        {
            finished = true;
        }
    }

    /**
     * True if game is finished.
     * @return
     */
    public boolean isFinished()
    {
        return finished;
    }

    /**
     * Utility method generating unique board key.
     * @return key in form
     */
    public String getKey()
    {
        String key = "";
        for( String[][] board : states)
        {
            for(int r = 0; r < board.length; r++)
            {
                for(int c = 0; c < board.length; c++)
                {
                    key += (board[r][c].trim().isEmpty() ? "." : board[r][c]);
                }
            }
            key += "|";
        }
        return key;
    }



}
