package com.ai.tictactoe.model.neuralnetwork.general;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * Activation functions and their derivatives
 */
public class Activation
{
    public static final Map<TransferFunction, Function<Double, Double>> activations = new HashMap<>();
    static
    {
        activations.put( TransferFunction.RELU,    net -> Math.max(0.0, net));
        activations.put( TransferFunction.SIGMOID, net -> 1 / (1 + Math.exp(-1.0 * net)));
        activations.put( TransferFunction.TANH,    net -> Math.tanh(net));
        activations.put( TransferFunction.SOFTMAX, net -> Math.exp(net)); // this is just nominator of softmax equation
    }

    public static final Map<TransferFunction, Function<Double, Double>> derivatives = new HashMap<>();
    static
    {
        derivatives.put( TransferFunction.SIGMOID, sig -> sig * (1.0 - sig));
        derivatives.put( TransferFunction.TANH, tanh -> 1.0 - Math.pow(tanh,2));
        derivatives.put( TransferFunction.RELU, rel -> rel <= 0 ? 0.0 : 1.0);
        derivatives.put( TransferFunction.SOFTMAX, s -> s * (1 - s));
    }



}
