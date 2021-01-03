package com.ai.tictactoe;

import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork;
import com.ai.tictactoe.util.BoardCell;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Wrapper class for neural network trained to play Tic-Tac-Toe game.
 *
 */
public class TicTacToeNetwork
{
    /** Neural network object **/
    NeuralNetwork ann;

    private final static Map<Integer, BoardCell> cellIndex2CellMap = new HashMap<>();

    /**
     * Default constructor
     */
    public TicTacToeNetwork()
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
    }

    /**
     * Init internal ANN object with data from file
     * @param annFileName
     */
    public void init(final String annFileName)
    {
        ann = NeuralNetwork.deserialize(annFileName);
    }

    /**
     * Predict next move for given board state
     * @param board - 2D string array representing Tic-Tac-Toe board
     * @return Cell object with row nad column
     */
    public BoardCell predictNextMove(final String[][] board)
    {
        //predict board fields preferences
        List<Double> predictedFields = ann.predict(board2Inputs_18(board));

        //pick most rated field from the list
        int highRatedFieldIndex = 0;
        double max = -1.0d;
        for (int i = 0; i < predictedFields.size(); i++)
        {
            if (max < predictedFields.get(i))
            {
                max = predictedFields.get(i);
                highRatedFieldIndex = i;
            }
        }

        //maps cell index to Tic-Tac-Toe board coordinates
        return cellIndex2CellMap.get(highRatedFieldIndex);
    }

    /**
     * Convert game board (2D string array) into list of 9 integer inputs where:
     * input value is 0 for an empty field
     * input value is 1 for "x"
     * input value is -1 for "o"
     * @param board
     * @return
     */
    public static List<Integer> board2Inputs_9(String[][] board)
    {
        int idx = 0;
        String value = "";
        final Integer[] boardInputs = new Integer[] {0,0,0,0,0,0,0,0,0};
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                value = board[i][j].trim();
                if(value.isEmpty())
                    boardInputs[idx++] = 0;
                else if( value.equals("x"))
                    boardInputs[idx++] = 1;
                else if(value.equals("o"))
                    boardInputs[idx++] = -1;
            }
        }
        return Arrays.asList(boardInputs);
    }

    /**
     *
     * @param board
     * @return
     */

    public static List<Integer> board2Inputs_18(String[][] board)
    {
        int idx = 0;
        final Integer[] boardInputs = new Integer[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("x") ? 1 : 0;
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("o") ? 1 : 0;
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
    public static List<Integer> board2Inputs_27(String[][] board)
    {
        int idx = 0;
        final Integer[] boardInputs = new Integer[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].trim().isEmpty() ? 1 : 0;
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("x") ? 1 : 0;
            }
        }
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                boardInputs[idx++] = board[i][j].equals("o") ? 1 : 0;
            }
        }
        return Arrays.asList(boardInputs);
    }

}
