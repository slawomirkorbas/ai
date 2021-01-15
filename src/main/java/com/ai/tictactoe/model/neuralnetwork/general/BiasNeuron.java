package com.ai.tictactoe.model.neuralnetwork.general;

public class BiasNeuron extends Neuron
{
    public BiasNeuron(String name, Double defaultValue)
    {
        super(name, null);
        this.outputValue = defaultValue;
        this.bias = true;
    }
}
