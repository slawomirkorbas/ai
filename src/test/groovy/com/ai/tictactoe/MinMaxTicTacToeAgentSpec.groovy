package com.ai.tictactoe

import com.ai.tictactoe.game.BoardCell
import com.ai.tictactoe.game.MinMaxTicTacToeAgent
import spock.lang.Specification
import spock.lang.Unroll

class MinMaxTicTacToeAgentSpec extends Specification
{
    @Unroll
    def "findBestMove: works as expected "()
    {
        given:
            MinMaxTicTacToeAgent agent = new MinMaxTicTacToeAgent("x")
            String[][] board = matrix.toArray()

        when:
            BoardCell move = agent.getNextMove(board)

        then:
            move.row == expRow
            move.col == expCol

        where:
            matrix          | expRow | expCol
            [["x","o"," "],
             [" ","o"," "],
             [" "," "," "]] | 2      | 1
            [["o"," ","x"],
             [" ","o"," "],
             [" "," "," "]] | 2      | 2
            [["o","x","x"],
             ["x","o"," "],
             ["o","o","x"]] | 1      | 2
            [["o","o","x"],
             [" "," ","o"],
             [" ","x","x"]] | 2      | 0
    }


    @Unroll
    def "findBestMove: returns null when the game is over"()
    {
        given:
            MinMaxTicTacToeAgent agent = new MinMaxTicTacToeAgent("x")
            String[][] board = matrix.toArray()

        expect:
            expResult == agent.getNextMove(board, )


        where:
            matrix          | expResult
            [["o","o","x"],
             ["x","x","o"],
             ["o","x","x"]] | null // board is full
            [["o","o","x"],
             ["o"," ","o"],
             ["x","x","x"]] | null // game is over: "x" has won
            [["o"," ","x"],
             ["o","o","x"],
             ["x"," ","o"]] | null // game is over: "o" has won
    }
}
