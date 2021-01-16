package com.ai.tictactoe.game;

/**
 * Class representing specific game state associated with the next move that has been done.
 */
public class GameState
{
    public String [][] board;
    public Move nextMove = null;

    public GameState(String[][] board)
    {
        this.board = board;
    }
}
