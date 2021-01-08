package com.ai.tictactoe

import com.ai.tictactoe.game.MinMaxTicTacToeAgent
import com.ai.tictactoe.game.RandomTicTacToeAgent
import com.ai.tictactoe.model.neuralnetwork.general.Activation
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork
import com.ai.tictactoe.model.neuralnetwork.general.WeightInitializationType
import com.ai.tictactoe.game.BoardCell
import spock.lang.Specification
import java.util.stream.Collectors

/**
 * Main test specification. Creates the Neural Network with given initial weights and learning rate.
 * Then train this ANN network with game data produced by two computer players "who" are using
 * Min-Max algorithm to predict next move.
 */
class TicTacToeMachineLearningSpec extends Specification
{
    /**
     * This supervised learning procedure is using two players: MinMaxTicTacToeAgent paying against RandomTicTacToeAgent
     * @return file with "adjusted" neural network object
     */
    def 'supervised learning using ANN with 1 output: train new tic-tac-toe network and serialize to file'()
    {
        given:
            Double learningRate = 0.1d
            final NeuralNetwork ann = new NeuralNetwork(learningRate, WeightInitializationType.XAVIER)
            ann.addLayer(18, "I", null, null)
            ann.addLayer(12, "H1", 0.01d, Activation.TANH)
            ann.addLayer(9, "H2", 0.01d, Activation.TANH)
            ann.addLayer(1 , "O" , 0.01d, Activation.RELU)
            int sampleNumber = 0
        and:
            MinMaxTicTacToeAgent minMaxAgent = new MinMaxTicTacToeAgent("x")
            RandomTicTacToeAgent randomAgent = new RandomTicTacToeAgent("o")

        when:
            5000.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
                while(true) {
                    BoardCell targetMoveX = minMaxAgent.getFirstNextMove(board)
                    List<Integer> input = TicTacToeEngine.board2Inputs_9(board)
                    if (targetMoveX == null) { // game finished
                        //ann.train(input, [Double.valueOf(-1)], ++sampleNumber)
                        break
                    }
                    Integer targetCellIndex = TicTacToeEngine.rowCol2CellIndexMap.get(targetMoveX.row + "_" + targetMoveX.col)
                    ann.train(input, [Double.valueOf(targetCellIndex)], ++sampleNumber)
                    board[targetMoveX.row][targetMoveX.col] = "x"

                    // player "O" turn...
                    BoardCell nextMoveO = randomAgent.getNextMove(board, )
                    if (nextMoveO == null) {
                        break
                    }
                    board[nextMoveO.row][nextMoveO.col] = "o"
                }
                System.out.println("Training for game executed: " + boardToString(board))
            }

        then:
            ann.serializeToFile()
            true
    }

    //@Ignore
    def 'supervised learning using ANN with 9 outputs: train new tic-tac-toe network and serialize to file'()
    {
        given:
            final Double learningRate = 0.1d
            final NeuralNetwork ann = new NeuralNetwork(learningRate, WeightInitializationType.XAVIER)
            ann.addLayer(18, "I" , null, null)
            ann.addLayer(15, "H1", 0.1d, Activation.TANH)
            ann.addLayer(12, "H2", 0.1d, Activation.TANH)
            ann.addLayer(9 , "O" , 0.1d, Activation.SIGMOID) // TODO apply softmax...
        and:
            MinMaxTicTacToeAgent playerX = new MinMaxTicTacToeAgent("x")
            RandomTicTacToeAgent randomAgent = new RandomTicTacToeAgent("o")
        and:
            int sampleNumber = 0, gameNo = 0
            Integer batchSize = 1000
            final Map<String, String[][]> uniqueGameStateMap = new HashMap<>()

        when:
            1000.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]

                while(true)
                {
                    List<BoardCell> targetMovesX = playerX.computeBestMoves(board, "x")
                    if (targetMovesX.size() > 0)
                    {
                        List<Integer> input = TicTacToeEngine.board2Inputs_18(board)
                        List<Double> targetOutput = cords2TargetOutput_9(targetMovesX)

                        // train network only for newly experienced game states
                       // final String gameStateKey = randomAgent.getBoardKey(board)
                       // if(null == uniqueGameStateMap.get(gameStateKey) )
                       // {
                            ann.train(input, targetOutput, ++sampleNumber)
                            //uniqueGameStateMap.put(gameStateKey, board)
                            //System.out.println(uniqueGameStateMap.size() + " game state trained: " + boardToString(board))
                       // }


                        // pick randomly best move from the list of best moves collected...
                        //int randomIndex = targetMovesX.size() == 1 ? 0 : (new Random()).nextInt(targetMovesX.size() - 1);
                        //board[targetMovesX.get(randomIndex).row][targetMovesX.get(randomIndex).col] = "x"
                        board[targetMovesX.get(0).row][targetMovesX.get(0).col] = "x"

                        // payer "O" turn...
                        BoardCell nextMoveO = randomAgent.getNextMove(board, "o")
                        if (nextMoveO == null) {
                            System.out.println(++gameNo + " Game training finished: " + boardToString(board))
                            break
                        }
                        board[nextMoveO.row][nextMoveO.col] = "o"
//                        List<BoardCell> targetMovesO = playerO.computeBestMoves(board, "o")
//                        if (targetMovesO.size() > 0)
//                        {
//                             int randomIndex = targetMovesO.size() == 1 ? 0 : (new Random()).nextInt(targetMovesO.size() - 1);
//                             board[targetMovesO.get(randomIndex).row][targetMovesO.get(randomIndex).col] = "o"
//                        }
                    }
                    else {
                        System.out.println(++gameNo + " Game training finished: " + boardToString(board))
                        break
                    }
                }
            }

        then:
            ann.serializeToFile()
            //ann.visualize()
            true

    }


    def 'Next predicted move should be different than the first one during a game'()
    {
        given:
            TicTacToeEngine ann =  new TicTacToeEngine();
            ann.init("neural-network-20210107-1527.ann")
        and:
            RandomTicTacToeAgent randomAgent = new RandomTicTacToeAgent("x")
            String[][] board = [["x", "o", " "], [" ", " ", " "], [" ", " ", " "]]

        when:
            BoardCell moveX = ann.predictNextMove(board)
            board[moveX.row][moveX.col] = "x"
        and:
            BoardCell moveO = randomAgent.getNextMove(board, "o")
            board[moveO.row][moveO.col] = "o"
        then:
            moveX != ann.predictNextMove(board)
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


    List<Double> cords2TargetOutput_9(final List<BoardCell> targetMoves)
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
