package com.ai.tictactoe.model.neuralnetwork.general;

public class InputLayer extends Layer
{

    public InputLayer(final int noOfNeurons, final String name)
    {
        super(noOfNeurons, name, null, null);
        this.inputLayer = true;
    }
}
