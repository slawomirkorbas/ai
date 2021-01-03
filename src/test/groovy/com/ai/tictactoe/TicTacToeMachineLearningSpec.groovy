package com.ai.tictactoe

import com.ai.tictactoe.model.neuralnetwork.general.ActivationFunction
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork
import com.ai.tictactoe.util.BoardCell
import spock.lang.Ignore
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

    NeuralNetwork createTicTacToeNetwork(inputSize, hiddenSize, outputSize)
    {
        def learningRate = 0.1d
        NeuralNetwork ann = new NeuralNetwork(learningRate)
        ann.addLayer(inputSize, "I", null, null)
        ann.addLayer(hiddenSize, "H", 0.1d, ActivationFunction.SIGMOID)
        ann.addLayer(outputSize, "O", 0.1d, ActivationFunction.SIGMOID)
        ann
    }

    //@Ignore
    def 'train new tic-tac-toe network and serialize to file'()
    {
        given:
            final NeuralNetwork ann = createTicTacToeNetwork(27, 36, 9)
            Integer[] targetMoveO
            Integer[] targetMoveX


        when:
            1000.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
                String[][] invertedBoard

                while(true)
                {
                    List<BoardCell> targetMovesX = playerX.computeBestMoves(board, "x")
                    if (targetMovesX.size() > 0) {
                        List<Integer> input = TicTacToeNetwork.board2Inputs(board)
                        List<Double> targetOutput = moveCords2TargetOutput(targetMovesX)

                        // perform learning iteration (prediction + back propagation)
                        //System.out.println("Train for move X. Board: " + boardToString(board))
                        ann.train(input, targetOutput)

                        // pick randomly best move from the list of best moves collected...
                        int randomIndex = targetMovesX.size() == 1 ? 0 : (new Random()).nextInt(targetMovesX.size() - 1);
                        board[targetMovesX.get(randomIndex).row][targetMovesX.get(randomIndex).col] = "x"

                        // payer "O" turn...
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


    List<Double> moveCords2TargetOutput(final List<BoardCell> targetMoves)
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
