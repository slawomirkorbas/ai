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
    public List<BoardStatus> states = new ArrayList<>();

    public boolean finished = false;

    public String key;

    public TicTacToeGame()
    {
        // add initial game state
        states.add(new BoardStatus(TicTacToeAgent.copyBoard(this.board)));
    }

    public void update(final GameResult result, BoardCell nextMove)
    {
        //store next move in previous game state
        states.get(states.size() -1).nextMove = nextMove;

        //adds new game state
        states.add(new BoardStatus(TicTacToeAgent.copyBoard(board)));
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
    public String generateKey()
    {
        this.key = "";
        for(BoardStatus boardStatus : states)
        {
            for(int r = 0; r < boardStatus.board.length; r++)
            {
                for(int c = 0; c < boardStatus.board.length; c++)
                {
                    key += (boardStatus.board[r][c].trim().isEmpty() ? "." : boardStatus.board[r][c]);
                }
            }
            key += "|";
        }
        return key;
    }



}
