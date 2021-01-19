package com.ai.tictactoe.game;

import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TicTacToe agent playing game by predicting moves using internal (trained) neural network.

 */
public class AnnTicTacToeAgent extends TicTacToeAgent
{
    /** Neural network object **/
    NeuralNetwork ann;

    /** Utility maps **/
    private final static Map<Integer, BoardCell> cellIndex2CellMap = new HashMap<>();
    private final static Map<String, Integer> rowCol2CellIndexMap  = new HashMap<>();
    static
    {
        cellIndex2CellMap.put(0, new BoardCell(0,0));
        cellIndex2CellMap.put(1, new BoardCell(0,1));
        cellIndex2CellMap.put(2, new BoardCell(0,2));
        cellIndex2CellMap.put(3, new BoardCell(1,0));
        cellIndex2CellMap.put(4, new BoardCell(1,1));
        cellIndex2CellMap.put(5, new BoardCell(1,2));
        cellIndex2CellMap.put(6, new BoardCell(2,0));
        cellIndex2CellMap.put(7, new BoardCell(2,1));
        cellIndex2CellMap.put(8, new BoardCell(2,2));

        rowCol2CellIndexMap.put( "0_0", 0);
        rowCol2CellIndexMap.put( "0_1", 1);
        rowCol2CellIndexMap.put( "0_2", 2);
        rowCol2CellIndexMap.put( "1_0", 3);
        rowCol2CellIndexMap.put( "1_1", 4);
        rowCol2CellIndexMap.put( "1_2", 5);
        rowCol2CellIndexMap.put( "2_0", 6);
        rowCol2CellIndexMap.put( "2_1", 7);
        rowCol2CellIndexMap.put( "2_2", 8);
    }

    /**
     * Default constructor
     */
    public AnnTicTacToeAgent(final String playAs)
    {
        super(playAs);
    }

    /**
     * Init internal ANN object with data from file
     * @param annFileName
     */
    public void init(final String annFileName)
    {
        ann = NeuralNetwork.deserialize(annFileName);
    }

    @Override
    public BoardCell getNextMove(final String[][] board)
    {
        if(matrixFull(board))
        {
            return null;
        }
        return predictNextMove(board);
    }

    /**
     * Predict next move for given board state
     * @param board - 2D string array representing Tic-Tac-Toe board
     * @return Cell object with row nad column
     */
    private BoardCell predictNextMove(final String[][] board)
    {
        final List<Double> inputs = inputVectorFromBoard(board, ann.getInputLayer().numberOfNeurons());
        final List<Double> outputVector = ann.predict(inputs);

        if(ann.isSingleOutput())
        {
            Integer cellIndex = (int)Math.round(outputVector.get(0));
            if(cellIndex < 0 || cellIndex > 8)
            {
                return null;
            }
            return cellIndex2CellMap.get(cellIndex);
        }
        else
        {
            //pick most rated field from the list
            Integer topRankedFieldIndex = null;
            double max = -10000000.0;
            for(int i = 0; i < outputVector.size(); i++)
            {
                BoardCell cell = cellIndex2CellMap.get(i); // get the highest ranked field freom output vector which is not occupied on the board
                if(max < outputVector.get(i) ) //&& board[cell.row][cell.col].trim().isEmpty())
                {
                    max = outputVector.get(i);
                    topRankedFieldIndex = i;
                }
            }

            //maps cell index to Tic-Tac-Toe board coordinates
            return cellIndex2CellMap.get(topRankedFieldIndex);
        }
    }

    /**
     * Format input vector from given game board based on input layer size.
     * @param board - game state board
     * @return list of input values
     */
    public static List<Double> inputVectorFromBoard(final String[][] board, int inputSize)
    {
        final List<Double> inputs;
        switch(inputSize)
        {
            case 9:
                inputs = board2Inputs_9(board);
                break;
            case 18:
                inputs = board2Inputs_18(board);
                break;
            case 27:
                inputs = board2Inputs_27(board);
                break;
            default:
                inputs = null;
        }
        return inputs;
    }


    /**
     * Convert game board (2D string array) into list of 9 integer inputs where:
     * input value is 0 for an empty field
     * input value is 1 for "x"
     * input value is -1 for "o"
     * @param board
     * @return
     */
    public static List<Double> board2Inputs_9(String[][] board)
    {
        int idx = 0;
        String value = "";
        final Double[] boardInputs = new Double[] {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                value = board[i][j].trim();
                if(value.isEmpty())
                    boardInputs[idx++] = 0.01;
                else if( value.equals("x"))
                    boardInputs[idx++] = 1.0;
                else if(value.equals("o"))
                    boardInputs[idx++] = -1.0;
            }
        }
        return Arrays.asList(boardInputs);
    }

    /**
     * Convert board game to input vector of size 18
     * @param board
     * @return
     */

    public static List<Double> board2Inputs_18(String[][] board)
    {
        int idx = 0;
        final Double[] boardInputs = new Double[] {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("x") ? 1.0 : 0.0;
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("o") ? 1.0 : 0.0;
            }
        }
        return Arrays.asList(boardInputs);
    }

    /**
     * Convert game board (2D string array) into list of 27 integer inputs where:
     * First 9 fields represent empty cells within the matrix. (1 is set if the field is empty)
     * Fields from 9-17 represent cells occupied by <code>x</code> within the matrix. (1 is set if the field is occupied by x)
     * Fields from 18-26 represent cells occupied by <code>o</code> within the matrix. (1 is set if the field is occupied by o)
     * @param board - 2D string array
     * @return list of inputs capable for making predictions by neural network
     */
    public static List<Double> board2Inputs_27(String[][] board)
    {
        int idx = 0;
        final Double[] boardInputs = new Double[] {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].trim().isEmpty() ? 1.0 : 0.0;
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("x") ? 1.0 : 0.0;
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("o") ? 1.0 : 0.0;
            }
        }
        return Arrays.asList(boardInputs);
    }

}
