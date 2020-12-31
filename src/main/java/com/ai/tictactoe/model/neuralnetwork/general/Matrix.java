package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.Data;

import java.io.Serializable;
import java.util.List;
import java.util.function.Function;

/**
 * Matrix class providing basic manipulation methods like add, subtract, multiply etc.
 */
@Data
public class Matrix implements Serializable
{
    private static final long serialVersionUID = 1L;

    int rows;
    int cols;
    Double[][] fields;

    /**
     * Default Matrix constructor. Initializes each field with 0.
     * @param rows
     * @param cols
     */
    public Matrix(int rows, int cols)
    {
        fields = new Double[rows][cols];
        this.rows = rows;
        this.cols = cols;
        for(int i=0; i < rows; i++)
        {
            for(int j=0; j<cols; j++)
            {
                fields[i][j] = 0.0;
            }
        }
    }

    /**
     * Constructor using input array as an argument
     * @param inputArray
     */
    public Matrix(double[][] inputArray)
    {
        rows = inputArray.length;
        cols = inputArray[0].length;
        fields = new Double[rows][cols];
        for(int i=0; i < rows; i++)
        {
            for(int j=0; j < cols; j++)
            {
                fields[i][j] = inputArray[i][j];
            }
        }
    }

    /**
     * Constructor using ArrayList as an argument
     * @param inputArrayList
     */
    public Matrix(List<List<Double>> inputArrayList)
    {
        rows = inputArrayList.size();
        cols = inputArrayList.get(0).size();
        fields = new Double[rows][cols];
        for(int i=0; i < rows; i++)
        {
            for(int j=0; j < cols; j++)
            {
                fields[i][j] = inputArrayList.get(i).get(j);
            }
        }
    }

    /**
     * Copy constructor.
     * @param inputMatrix
     */
    public Matrix(Matrix inputMatrix)
    {
        fields = new Double[inputMatrix.rows][inputMatrix.cols];
        rows = inputMatrix.rows;
        cols = inputMatrix.cols;
        for(int i=0; i < rows; i++)
        {
            for(int j=0; j < cols; j++)
            {
                fields[i][j] = inputMatrix.fields[i][j];
            }
        }
    }

    @Override
    public int hashCode()
    {
        int result = (int) (rows ^ (cols >>> 32));
        result = 31 * result + fields.hashCode();
        return result;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null) return false;
        if (this.getClass() != o.getClass()) return false;

        if( cols!= ((Matrix)o).cols || rows != ((Matrix)o).rows)
        {
            return false;
        }

        for(int i=0; (i < rows); i++)
        {
            for(int j=0; (j < cols); j++)
            {
                if((Math.round(fields[i][j]) != Math.round(((Matrix)o).fields[i][j])))
                {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Adds scalar value to each matrix i,j field
     * @param scalar
     */
    public void add(double scalar)
    {
        for(int i=0; i < rows; i++)
        {
            for(int j=0; j < cols; j++)
            {
                fields[i][j] += scalar;
            }

        }
    }

    /**
     * Adds each value of corresponding [i][j] field in matrix to [i][j] field of this matrix.
     *  Each of them have to be of the same shape.
     * @param mat
     */
    public void add(final Matrix mat)
    {
        if( cols!= mat.cols || rows != mat.rows)
        {
            System.out.println("Cannot add matrices of different shapes.");
            return;
        }

        for(int i=0; i<rows; i++)
        {
            for(int j=0; j < cols; j++)
            {
                fields[i][j] += mat.fields[i][j];
            }
        }
    }

    /**
     * Subtracts two matrices. Each of them have to be of the same shape.
     * @param mat
     */
    public void subtract(final Matrix mat)
    {
        if( cols != mat.cols || rows != mat.rows)
        {
            System.out.println("Cannot subtract matrices of different shapes.");
            return;
        }

        for(int i=0; i < this.rows; i++)
        {
            for(int j=0; j < this.cols; j++)
            {
                fields[i][j] -= mat.fields[i][j];
            }
        }
    }

    /**
     * Multiplies the matrix by the scalar value.
     * @param multiplier
     */
    public void multiply(final double multiplier)
    {
        for(int i=0; i < rows; i++)
        {
            for(int j=0; j < cols; j++)
            {
                fields[i][j] *= multiplier;
            }
        }
    }

    /**
     * Multiplies the matrix by the given matrix
     * @param mat
     */
    public void multiply(final Matrix mat)
    {
        for(int i=0; i<mat.rows; i++)
        {
            for(int j=0; j<mat.cols; j++)
            {
                fields[i][j] *= mat.fields[i][j];
            }
        }

    }


    /**
     * Creates transposition of the matrix.
     */
    public void transpose()
    {
        for(int i=0; i< this.rows; i++)
        {
            for(int j=0; j < this.cols; j++)
            {
                fields[j][i] = fields[i][j];
            }
        }
    }

    /**
     * Applies specific hyperbolic function to each element of the matrix
     * @param fun
     */
    public void applyFunction(Function<Double, Double> fun)
    {
        for(int i=0; i< rows; i++)
        {
            for(int j=0; j< cols; j++)
            {
                fields[i][j] = fun.apply(fields[i][j]);
            }
        }
    }
}
