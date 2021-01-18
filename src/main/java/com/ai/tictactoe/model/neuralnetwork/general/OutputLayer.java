package com.ai.tictactoe.model.neuralnetwork.general;


public class OutputLayer extends Layer
{

    /** Loss function along with its derivative formula. This is used only for output layer **/
    CostFunction costFunction;

    /**
     * Default constructor
     *
     * @param costFunction
     */
    public OutputLayer(final int noOfNeurons, final String name, final Double initialWeight, TransferFunction transferFunc, CostFunction costFunction)
    {
        super(noOfNeurons, name, initialWeight, transferFunc);
        this.costFunction = costFunction;
        this.outputLayer = true;
    }

}
