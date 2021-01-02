package com.ai.tictactoe

import com.ai.tictactoe.model.neuralnetwork.general.ActivationFunction
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork
import spock.lang.Ignore
import spock.lang.Specification

/**
 * Main test specification. Creates the Neural Network with given initial weights and learning rate.
 * Then train this ANN network with game data produced by two computer players "who" are using
 * Min-Max algorithm to predict next move.
 */
class TicTacToeMachineLearningSpec extends Specification
{
    MinMaxTicTacToeEngine playerX = new MinMaxTicTacToeEngine()
    MinMaxTicTacToeEngine playerO = new MinMaxTicTacToeEngine()


    //@Ignore
    def 'train new tic-tac-toe network and serialize to file'()
    {
        given:
            final NeuralNetwork ann = createTicTacToeNetwork(27, 14, 9)
            Integer[] targetMoveO
            Integer[] targetMoveX


        when:
            100.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
                String[][] invertedBoard

                while(true)
                {
                    targetMoveX = playerX.findBestMove(board, "x")
                    if (targetMoveX != null)
                    {
                        List<Integer> input = TicTacToeNetwork.board2Inputs(board)
                        List<Double> targetOutput = moveCordsToTargetOutput(board, targetMoveX)

                        // perform learning iteration (prediction + back propagation)
                        System.out.println("Train for move X. Board: " + boardToString(board))
                        ann.train(input, targetOutput)

                        board[targetMoveX[0]][targetMoveX[1]] = "x"
                        targetMoveO = playerO.findBestMove(board, "o")
                        if (targetMoveO != null)
                        {
                            // perform learning iteration (prediction + back propagation)
                            // for the inverted board
                            //invertedBoard = invertBoard(board)
                            //input = TicTacToeNetwork.board2Inputs(invertedBoard)
                            //targetOutput = moveCordsToTargetOutput(board, targetMoveO)
                            //System.out.println("Train for move X. Board: " + boardToString(invertedBoard))
                            //ann.train(input, targetOutput)
                            board[targetMoveO[0]][targetMoveO[1]] = "o"
                            continue
                        }
                    }
                    break
                }
            }

        then:
            ann.serializeToFile()
            true

    }

    String boardToString(String[][]board)
    {
        String output = ""
        for(int i=0; i<board.length; i++) {
            for (int j = 0; j < board.length; j++) {
                output += board[i][j] == " " ? "_" : board[i][j]
            }
        }
        return output
    }

    String[][] invertBoard(String[][] board)
    {
        String[][] invertedBoard = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                if(!board[i][j].trim().isEmpty())
                    invertedBoard[i][j] = board[i][j] == "x" ? "o" : "x"
            }
        }
        invertedBoard;
    }


    List<Double> moveCordsToTargetOutput(String[][] board, Integer[] expectedMove)
    {
        List<Double> targetOutput = [0.0d, 0.0d, 0.0d, 0.0d, 0.0d, 0.0d, 0.0d, 0.0d, 0.0d]
        int index = 0
        for(int row=0; row<board.length; row++)
        {
            for(int col=0; col<board.length; col++)
            {
                if(expectedMove[0] == row && expectedMove[1] == col)
                {
                    targetOutput[index] = 1.0d
                }
                index++
            }
        }
        return targetOutput
    }


    NeuralNetwork createTicTacToeNetwork(inputSize, hiddenSize, outputSize)
    {
        def learningRate = 0.5d
        NeuralNetwork ann = new NeuralNetwork(learningRate)
        ann.addLayer(inputSize, "I", null, null)
        ann.addLayer(hiddenSize, "H", 0.0d, ActivationFunction.SIGMOID)
        ann.addLayer(outputSize, "O", 0.0d, ActivationFunction.SIGMOID)
        ann
    }


}
