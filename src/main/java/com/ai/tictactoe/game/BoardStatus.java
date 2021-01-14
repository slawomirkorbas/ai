package com.ai.tictactoe.game;

public class BoardStatus
{
    public String [][] board;
    public BoardCell nextMove = null;

    public BoardStatus(String[][] board)
    {
        this.board = board;
    }
}
