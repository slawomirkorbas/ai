package com.ai.tictactoe

import com.ai.tictactoe.model.neuralnetwork.general.ActivationFunction
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork
import spock.lang.Specification

class TicTacToeMachineLearningSpec extends Specification
{
    MinMaxTicTacToeEngine playerX = new MinMaxTicTacToeEngine()
    MinMaxTicTacToeEngine playerO = new MinMaxTicTacToeEngine()


    def 'train new tic-tac-toe network and serialize to file'()
    {
        given:
            NeuralNetwork ann = createTicTacToeNetwork(27, 14, 9)
            Integer[] moveO
            Integer[] targetMoveX


        when:
            100.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]

                while(true)
                {
                    targetMoveX = playerX.findBestMove(board, "x")
                    if (targetMoveX != null)
                    {
                        List<Integer> input = board2Inputs(board)
                        List<Double> targetOutput = moveCordsToTargetOutput(board, targetMoveX)

                        // perform learn iteration (prediction + back propagation)
                        ann.train(input, targetOutput)

                        board[targetMoveX[0]][targetMoveX[1]] = "x"
                        moveO = playerO.findBestMove(board, "o")
                        if (moveO != null) {
                            board[moveO[0]][moveO[1]] = "o"
                            continue
                        }
                    }
                    break
                }
            }

        then:
            true //TODO save ann to file...

    }


    List<Integer> board2Inputs(String[][] board)
    {
        int idx = 0;
        List<Integer> boardInputs = new ArrayList<>(27)
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].trim().isEmpty() ? 1 : 0
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("x") ? 1 : 0
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("o") ? 1 : 0
            }
        }
        return boardInputs
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
        def learningRate = 0.2d
        NeuralNetwork ann = new NeuralNetwork(learningRate)
        ann.addLayer(inputSize, "I", null, null)
        ann.addLayer(hiddenSize, "H", 0.1d, ActivationFunction.SIGMOID)
        ann.addLayer(outputSize, "O", 0.1d, ActivationFunction.SIGMOID)
        ann
    }


}
