package com.ai.tictactoe

import com.ai.tictactoe.model.neuralnetwork.general.ActivationFunction
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork
import com.ai.tictactoe.util.BoardCell
import spock.lang.Specification

import java.util.stream.Collectors

/**
 * Main test specification. Creates the Neural Network with given initial weights and learning rate.
 * Then train this ANN network with game data produced by two computer players "who" are using
 * Min-Max algorithm to predict next move.
 */
class TicTacToeMachineLearningSpec extends Specification
{
    MinMaxTicTacToeEngine playerX = new MinMaxTicTacToeEngine()
    MinMaxTicTacToeEngine playerO = new MinMaxTicTacToeEngine()

    NeuralNetwork createTicTacToeNetwork()
    {
        Double learningRate = 0.1d
        NeuralNetwork ann = new NeuralNetwork(learningRate)
        ann.addLayer(18 , "I", null, null)
        ann.addLayer(12, "H1", 0.1d, ActivationFunction.RELU)
        // ann.addLayer(12, "H2", 0.01d, ActivationFunction.TANH)
        ann.addLayer(9 , "O", 0.1d, ActivationFunction.SIGMOID)
        ann
    }

    //@Ignore
    def 'train new tic-tac-toe network and serialize to file'()
    {
        given:
            final NeuralNetwork ann = createTicTacToeNetwork()
            int sampleNumber = 0
            Integer batchSize = 100

        when:
            batchSize.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
                String[][] invertedBoard

                while(true)
                {
                    List<BoardCell> targetMovesX = playerX.computeBestMoves(board, "x")
                    if (targetMovesX.size() > 0)
                    {
                        List<Integer> input = TicTacToeNetwork.board2Inputs_18(board)
                        List<Double> targetOutput = cords2TargetOutput(targetMovesX)

                        // perform learning iteration (prediction + back propagation)
                        //System.out.println("Train for move X. Board: " + boardToString(board))
                        ann.train(input, targetOutput, ++sampleNumber)

                        // pick randomly best move from the list of best moves collected...
                        int randomIndex = targetMovesX.size() == 1 ? 0 : (new Random()).nextInt(targetMovesX.size() - 1);
                        board[targetMovesX.get(randomIndex).row][targetMovesX.get(randomIndex).col] = "x"

                        // payer "O" turn...
                        List<BoardCell> targetMovesO = playerO.computeBestMoves(board, "o")
                        if (targetMovesO.size() > 0)
                        {
                            // perform learning iteration (prediction + back propagation) for the inverted board
                            invertedBoard = invertBoard(board)
                            input = TicTacToeNetwork.board2Inputs_18(invertedBoard)
                            targetOutput = cords2TargetOutput(targetMovesO)
                            //System.out.println("Train for move X. Board: " + boardToString(invertedBoard))
                            ann.train(input, targetOutput, ++sampleNumber)

                            randomIndex = targetMovesO.size() == 1 ? 0 : (new Random()).nextInt(targetMovesO.size() - 1);
                            board[targetMovesO.get(randomIndex).row][targetMovesO.get(randomIndex).col] = "o"
                        }
                    }
                    break
                }
            }

        then:
            ann.serializeToFile()
            //ann.visualize()
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


    List<Double> cords2TargetOutput(final List<BoardCell> targetMoves)
    {
        Map<String,Integer[]> bestMoves = targetMoves.stream().collect(Collectors.toMap(c -> c.row + "_" + c.col, c -> c ))
        List<Double> targetOutput = new ArrayList<>()
        for(int row=0; row< 3; row++) {
            for(int col=0; col< 3; col++) {
                targetOutput.add(bestMoves.get(row + "_" + col) != null ? 1.0d : 0.0d)
            }
        }
        return targetOutput
    }

}
