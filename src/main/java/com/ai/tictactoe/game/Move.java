package com.ai.tictactoe.game;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * Class representing the next move in game
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class Move
{
    /** Coordinates of the next filed within the board **/
    BoardCell cell;

    /** Figure used for the next move: "x" or "o" **/
    String figure;
}
