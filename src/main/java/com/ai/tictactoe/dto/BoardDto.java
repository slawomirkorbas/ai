package com.ai.tictactoe.dto;

import com.ai.tictactoe.game.GameResult;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;


@Data
@NoArgsConstructor
public class BoardDto implements Serializable
{
    private static final long serialVersionUID = 1L;

    /** result of the game **/
    public GameResult result = GameResult.CONTINUE;

    /** matrix of fields representing the tic-tac-toe board**/
    public String[][] board = new String[][] { {" ", " ", " "}, {" ", " ", " "}, {" ", " ", " "} };
}
