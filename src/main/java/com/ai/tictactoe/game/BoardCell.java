package com.ai.tictactoe.game;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * Cell on the Tic-Tac-Toe board identified by row and column index
 */
@Data
@AllArgsConstructor
public class BoardCell implements Serializable
{
    public Integer row;
    public Integer col;
}
