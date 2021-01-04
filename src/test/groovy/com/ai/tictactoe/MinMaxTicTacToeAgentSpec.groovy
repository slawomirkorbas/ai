package com.ai.tictactoe

import com.ai.tictactoe.agent.MinMaxTicTacToeAgent
import com.ai.tictactoe.util.BoardCell
import spock.lang.Specification
import spock.lang.Unroll

class MinMaxTicTacToeAgentSpec extends Specification
{
    @Unroll
    def "findBestMove: works as expected "()
    {
        given:
            MinMaxTicTacToeAgent engine = new MinMaxTicTacToeAgent()
            String[][] board = matrix.toArray()

        when:
            BoardCell move = engine.getNextMove(board, "x")

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
            MinMaxTicTacToeAgent engine = new MinMaxTicTacToeAgent()
            String[][] board = matrix.toArray()

        expect:
            expResult == engine.getNextMove(board, "x")


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




    def "generateGames"()
    {
        given:
            MinMaxTicTacToeAgent minMaxEngine01 = new MinMaxTicTacToeAgent();
            MinMaxTicTacToeAgent minMaxEngine02 = new MinMaxTicTacToeAgent();

        when:
            for(int r=0; r<3; r++)
            {
                for(int c=0; c<3; c++)
                {
                    String[][] board = [[" "," "," "],[" "," "," "], [" "," "," "]]
                    board[r][c] = "o"
                    while(true)
                    {
                        BoardCell move = minMaxEngine01.getNextMove(board, "x")
                        if (move != null) {
                            board[move.row][move.col] = "x"
                            move = minMaxEngine02.getNextMove(board, "o")
                        }
                        if (move != null) {
                            board[move.row][move.col] = "o"
                        }
                        if(move == null )
                        {
                            //game is complete - save the board....
                            board;
                            break;
                        }
                    }

                }
            }

        then:
            true
    }

}
