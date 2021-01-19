package com.ai.tictactoe

import com.ai.tictactoe.game.AnnTicTacToeAgent
import com.ai.tictactoe.game.GameState
import com.ai.tictactoe.game.GameResult
import com.ai.tictactoe.game.MinMaxTicTacToeAgent
import com.ai.tictactoe.game.RandomTicTacToeAgent
import com.ai.tictactoe.game.TicTacToeAgent
import com.ai.tictactoe.game.TicTacToeGame
import com.ai.tictactoe.model.neuralnetwork.general.CostFunction
import com.ai.tictactoe.model.neuralnetwork.general.DataSet
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetworkFactory
import com.ai.tictactoe.model.neuralnetwork.general.TransferFunction
import com.ai.tictactoe.model.neuralnetwork.general.WeightInitType
import com.ai.tictactoe.game.BoardCell
import com.fasterxml.jackson.databind.ObjectMapper
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
    static final NeuralNetworkFactory nnf = new NeuralNetworkFactory()

    /**
     * Machine learning based on game states stored in JSON file.
     */
    def 'File based supervised learning using JSON file with stored tic-tac-toe plays'()
    {
        given:
            NeuralNetwork ann = nnf.build()
                    .input(27, "I")
                    .hidden(36, "H1", 0.1d, TransferFunction.TANH)
                    //.hidden(12, "H2", 0.1d, TransferFunction.TANH)
                    .output(9 , "O" , 0.1d, TransferFunction.SOFTMAX, CostFunction.CROSS_ENTROPY)
                    .initialize(WeightInitType.RANDOM)
        and:
            int totalEpochs = 0
            // create datasets eg. 200 games per each chunk
            List<DataSet> dataSets_1 = prepareDatasetsFromFile("20210118-2323-game-batch-mmX-vs-rndO-315.json", 100,
                    ann.getInputLayer().numberOfNeurons(),
                    ann.getOutputLayer().numberOfNeurons())
            List<DataSet> dataSets_2 = prepareDatasetsFromFile("20210118-2323-game-batch-mmX-vs-rndO-315.json", 100,
                    ann.getInputLayer().numberOfNeurons(),
                    ann.getOutputLayer().numberOfNeurons())



        when:
            dataSets_1.forEach( dataSet -> {
                totalEpochs += ann.train(dataSet)
            })
        and:
            dataSets_2.forEach( dataSet -> {
                totalEpochs += ann.train(dataSet)
            })


        then:  // save trained network to file
            ann.serializeToFile(totalEpochs, dataSets_1.size() + dataSets_2.size() )
            true
    }

    /**
     * Prepare list of data sets compatible wih NeuralNetwork
     * @param jsonGamesFile - JSON file storing games
     * @param chunkSize - chink size
     * @param inputSize - size of input vector (eg. number of input neurons in Neural Network)
     * @param targetSize - size of target vector
     *
     * @return collection of datasets
     */
    private List<DataSet> prepareDatasetsFromFile(final String jsonGamesFileName,
                                          int chunkSize, int inputSize, int targetSize)
    {
        ObjectMapper mapper = new ObjectMapper()
        final File jsonGamesFile = new File(jsonGamesFileName)
        List<TicTacToeGame> gameList = mapper.readValue(jsonGamesFile, List<TicTacToeGame>.class)

        List<DataSet> dataSets = new ArrayList<>()
        int chunkIndex = 0
        while(true)
        {
            List gameSubList = getChunkOf(gameList, chunkIndex++, chunkSize)
            if(gameSubList == null) {
                break
            }
            for( TicTacToeGame g : gameSubList) {
                DataSet ds = new DataSet()
                String effectiveFigure = g.whoWon == null ? "x" : g.whoWon
                for (GameState gameState : g.states)
                {
                    if (gameState.nextMove != null && gameState.nextMove.figure.equals(effectiveFigure))
                    {
                        ds.addExample(AnnTicTacToeAgent.inputVectorFromBoard(gameState.board, inputSize),
                                cords2TargetVector(gameState.nextMove.cell, targetSize))
                    }
                }
                if (ds.size() > 0) {
                    dataSets.add(ds)
                }
            }
        }
        return dataSets
    }

    /**
     * Return sublist of given list having
     * @param inputList
     * @param index
     * @param chunkSize
     * @return sublist
     */
    private List getChunkOf(List inputList, int index, int chunkSize)
    {
        int start = index * chunkSize;
        int end = Math.min(start + chunkSize, inputList.size());
        if(start < end)
        {
            return new ArrayList<>(inputList.subList(start, end))
        }
        return null
    }


    def 'Next predicted move should be different than the first one during a game'()
    {
        given:
            AnnTicTacToeAgent annTicTacToeAgent =  new AnnTicTacToeAgent("x");
            annTicTacToeAgent.init("net-27-36-9-20210119-0853-batch-size-630-epochs-1353.ann")
        and:
            RandomTicTacToeAgent randomAgent = new RandomTicTacToeAgent("o")
            String[][] board = [["x", "o", " "], [" ", "o", " "], [" ", " ", " "]]

        when:
            BoardCell moveX = annTicTacToeAgent.getNextMove(board)
            board[moveX.row][moveX.col] = "x"
        and:
            randomAgent.doMove(board)
        then:
            moveX != annTicTacToeAgent.predictNextMove(board)
    }


    def 'Performance test of specific ANN TicTacToe Agent against RandomTicTacToeAgent'()
    {
        given:
            RandomTicTacToeAgent randomPlayer = new RandomTicTacToeAgent("o")
            AnnTicTacToeAgent annPlayer = new AnnTicTacToeAgent("x")
            annPlayer.init("net-27-36-9-20210119-0853-batch-size-630-epochs-1353.ann")
        and:
            int wins = 0, draws = 0, losses = 0
            Integer gamesTotal = 10000

        when:
            gamesTotal.times {
                String[][] board = emptyBoard()
                GameResult result
                TicTacToeAgent startingPlayer = annPlayer
                TicTacToeAgent followingPlayer = randomPlayer
                while(true)
                {
                    result = startingPlayer.doMove(board)
                    if(GameResult.CONTINUE != result && startingPlayer == annPlayer)
                    {
                        wins = result == GameResult.WIN ? wins + 1 : wins
                        draws = result == GameResult.DRAW ? draws + 1 : draws
                        break
                    }
                    result = followingPlayer.doMove(board)
                    if(GameResult.CONTINUE != result && followingPlayer == annPlayer)
                    {
                        wins = result == GameResult.WIN ? wins + 1 : wins
                        draws = result == GameResult.DRAW ? draws + 1 : draws
                        break
                    }
                }
                System.out.println("Game result: " + result)
                startingPlayer  = startingPlayer == annPlayer ? randomPlayer : annPlayer // swap players
                followingPlayer = startingPlayer == annPlayer ? randomPlayer : annPlayer
            }
        then:
            Double performance = (wins + draws)/gamesTotal * 100
            performance >= 80
            System.out.println("Overall performance of AnnTicTacToe player is: " + performance + "%")
    }


    /**
     * This supervised learning procedure is using two players: MinMaxTicTacToeAgent paying against RandomTicTacToeAgent
     * @return file with "adjusted" neural network object
     */
    def 'Dynamic supervised learning using ANN having vector SOFTMAX output : train new tic-tac-toe network and serialize to file'()
    {
        given:
        NeuralNetwork ann = nnf.build()
                .input(18, "I")
                .hidden(15, "H1", 0.01d, TransferFunction.TANH)
                .hidden(12, "H2", 0.01d, TransferFunction.TANH)
                .output(9 , "O" , 0.01d, TransferFunction.SOFTMAX, CostFunction.CROSS_ENTROPY)
                .initialize(WeightInitType.XAVIER)
        and:
            int sampleNumber = 0
            MinMaxTicTacToeAgent minMaxAgent = new MinMaxTicTacToeAgent("x")
            RandomTicTacToeAgent randomAgent = new RandomTicTacToeAgent("o")

        when:
        10.times {
            playGamesAndTrain(emptyBoard(), ann, minMaxAgent, randomAgent, sampleNumber)
        }

        then:
            // smoke test for SOFTMAX results
//            String[][] board = [["x", "o", " "], [" ", " ", " "], [" ", " ", " "]]
//            List<Double> output = ann.predict(AnnTicTacToeAgent.board2Inputs_18(board))
//            output.forEach( v -> {
//                System.out.print(v + ", ")
//                v >= 0.0d && v <= 1.0d // all values should be between 0 and 1
//            })
//        and:
            ann.serializeToFile(10, sampleNumber)
            true
    }

    String[][] emptyBoard()
    {
        [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
    }

    Integer playGamesAndTrain(final String[][] board, final NeuralNetwork ann,
                              MinMaxTicTacToeAgent agentX, TicTacToeAgent agentO, Integer sample)
    {
        List<BoardCell> targetMovesX = agentX.computeBestMoves(board, "x")
        List<Integer> input = AnnTicTacToeAgent.board2Inputs_18(board)
        List<Double> targetOutput = cords2TargetOutput_9multi(targetMovesX)

        // train the network
        ann.train(new DataSet(input,targetOutput))

        for(BoardCell moveX : targetMovesX)
        {
            String[][] tmp = agentO.copyBoard(board)
            tmp[moveX.row][moveX.col] = "x"
            if(GameResult.CONTINUE == agentO.doMove(tmp))
            {
                playGamesAndTrain(tmp, ann, agentX, agentO, sample)
            }
            else
            {
                System.out.println("Training for game executed: " + boardToString(tmp))
                sample++
            }
        }
        return sample
    }

    /**
     * This supervised learning procedure is using two players: MinMaxTicTacToeAgent paying against RandomTicTacToeAgent
     * @return file with "adjusted" neural network object
     */
    @Ignore
    def 'supervised learning using ANN with 1 output: train new tic-tac-toe network and serialize to file'()
    {
        given:
            NeuralNetwork ann = nnf.build()
                    .input(18, "I", )
                    .hidden(12, "H1", 0.01d, TransferFunction.TANH)
                    .hidden(9, "H2", 0.01d, TransferFunction.TANH)
                    .output(1 , "O" , 0.01d, TransferFunction.RELU, CostFunction.MSE)
                    .learningRate(0.1d)
                    .initialize(WeightInitType.XAVIER)
        and:
            int sampleNumber = 0
            MinMaxTicTacToeAgent minMaxAgent = new MinMaxTicTacToeAgent("x")
            RandomTicTacToeAgent randomAgent = new RandomTicTacToeAgent("o")
        and:
            Integer epochSize = 5000

        when:
            epochSize.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
                while(true) {
                    BoardCell targetMoveX = minMaxAgent.getFirstNextMove(board)
                    List<Integer> input = AnnTicTacToeAgent.board2Inputs_9(board)
                    if (targetMoveX == null) { // game finished
                        //ann.train(input, [Double.valueOf(-1)], ++sampleNumber)
                        break
                    }
                    Integer targetCellIndex = AnnTicTacToeAgent.rowCol2CellIndexMap.get(targetMoveX.row + "_" + targetMoveX.col)
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
            ann.serializeToFile(epochSize, sampleNumber)
            true
    }

    @Ignore
    def 'supervised learning using ANN with 9 outputs: train new tic-tac-toe network and serialize to file'()
    {
        given:
            NeuralNetwork ann = nnf.build()
                    .input(18, "I" , null, null)
                    .hidden(15, "H1", 0.1d, TransferFunction.TANH)
                    .hidden(12, "H2", 0.1d, TransferFunction.TANH)
                    .output(9 , "O" , 0.1d, TransferFunction.SIGMOID,  CostFunction.MSE)
                    .learningRate(0.1d)
                    .initialize(WeightInitType.XAVIER)
        and:
            MinMaxTicTacToeAgent playerX = new MinMaxTicTacToeAgent("x")
            RandomTicTacToeAgent randomAgent = new RandomTicTacToeAgent("o")
        and:
            int sampleNumber = 0, gameNo = 0
            Integer batchSize = 1000
            final Map<String, String[][]> uniqueGameStateMap = new HashMap<>()
            Integer epochSize = 1000

        when:
            epochSize.times {
                String[][] board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]

                while(true)
                {
                    List<BoardCell> targetMovesX = playerX.computeBestMoves(board, "x")
                    if (targetMovesX.size() > 0)
                    {
                        List<Integer> input = AnnTicTacToeAgent.board2Inputs_18(board)
                        List<Double> targetOutput = cords2TargetOutput_9multi(targetMovesX)

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
            ann.serializeToFile(epochSize, sampleNumber)
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
                    invertedBoard[i][j] = (board[i][j] == "x" ? "o" : "x")
            }
        }
        invertedBoard;
    }


    List<Double> cords2TargetOutput_9multi(final List<BoardCell> targetMoves)
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

    private List<Double> cords2TargetVector(final BoardCell targetCell, int targetSize)
    {
        if(targetSize == 1)
        {
            Integer targetCellIndex = AnnTicTacToeAgent.rowCol2CellIndexMap.get(targetCell.row + "_" + targetCell.col)
            return [Double.valueOf(targetCellIndex)]
        }
        else if (targetSize == 9)
        {
            return cords2TargetOutput_9(targetCell)
        }
        return null
    }

    List<Double> cords2TargetOutput_9(final BoardCell targetMove)
    {
        List<Double> targetOutput = new ArrayList<>()
        for(int row=0; row< 3; row++) {
            for(int col=0; col< 3; col++) {
                targetOutput.add((targetMove.row == row && targetMove.col == col) ? 1.0d : 0.0d)
            }
        }
        return targetOutput
    }


}
