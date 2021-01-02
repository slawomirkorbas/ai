package com.ai.tictactoe.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;


@Data
@NoArgsConstructor
public class BoardDto implements Serializable
{
    private static final long serialVersionUID = 1L;

    /** result of the game **/
    public Integer result = null;

    /** matrix of fields representing the tic-tac-toe board**/
    public String[][] board = new String[][] { {" ", " ", " "}, {" ", " ", " "}, {" ", " ", " "} };
}
