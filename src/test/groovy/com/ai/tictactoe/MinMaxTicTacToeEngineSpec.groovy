package com.ai.tictactoe

import spock.lang.Specification
import spock.lang.Unroll

class MinMaxTicTacToeEngineSpec extends Specification
{
    @Unroll
    def "findBestMove: works as expected "()
    {
        given:
            MinMaxTicTacToeEngine engine = new MinMaxTicTacToeEngine()
            String[][] board = matrix.toArray()

        when:
            Integer[] move = engine.findBestMove(board, "x")

        then:
            move[0] == expRow
            move[1] == expCol

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
            MinMaxTicTacToeEngine engine = new MinMaxTicTacToeEngine()
            String[][] board = matrix.toArray()

        expect:
            expResult == engine.findBestMove(board, "x")


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
            MinMaxTicTacToeEngine minMaxEngine01 = new MinMaxTicTacToeEngine();
            MinMaxTicTacToeEngine minMaxEngine02 = new MinMaxTicTacToeEngine();

        when:
            for(int r=0; r<3; r++)
            {
                for(int c=0; c<3; c++)
                {
                    String[][] board = [[" "," "," "],[" "," "," "], [" "," "," "]]
                    board[r][c] = "o"
                    while(true)
                    {
                        Integer[] move = minMaxEngine01.findBestMove(board, "x")
                        if (move != null) {
                            board[move[0]][move[1]] = "x"
                            move = minMaxEngine02.findBestMove(board, "o")
                        }
                        if (move != null) {
                            board[move[0]][move[1]] = "o"
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
