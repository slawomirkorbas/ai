package com.ai.tictactoe.model.neuralnetwork.general;


public class OutputLayer extends Layer
{
    /** Loss function along with its derivative formula. This is used only for output layer **/
    LossFunction lossFunction;

    /**
     * Default constructor
     * @param lossFunction
     */
    public OutputLayer(final int noOfNeurons,
                       final String name,
                       final Double initialWeight,
                       ActivationFunction function,
                       LossFunction lossFunction)
    {
        super(noOfNeurons, name, initialWeight, function);
        this.lossFunction = lossFunction;
    }

    boolean isOutputLayer()
    {
        return true;
    }
}