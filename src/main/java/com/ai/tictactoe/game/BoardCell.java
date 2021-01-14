package com.ai.tictactoe.game;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * Cell on the Tic-Tac-Toe board identified by row and column index
 */
public class BoardCell implements Serializable
{
    public Integer row;
    public Integer col;

    public BoardCell()
    {

    }

    public BoardCell(Integer row, Integer col)
    {
        this.row = row;
        this.col = col;
    }
}
