package com.ai.tictactoe.model.neuralnetwork.general;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * Activation functions and their derivatives
 */
public class Activation
{
    public static ActivationFunction RELU       = net -> Math.max(0.0, net);
    public static ActivationFunction LEAKY_RELU = net -> net <= 0 ? 0.01 * net : net;
    public static ActivationFunction SIGMOID    = net -> 1 / (1 + Math.exp(-1.0 * net));
    public static ActivationFunction TANH       = net -> Math.tanh(net);
    public static ActivationFunction SOFTMAX    = net -> net; // this is just a constant telling that SOFTMAX with more parameters should be used

    public static final Map<ActivationFunction, Function<Double, Double>> derivatives = new HashMap<>();
    static
    {
        derivatives.put( Activation.SIGMOID, sig -> sig * (1.0 - sig));
        derivatives.put( Activation.TANH, tanh -> 1.0 - Math.pow(tanh,2));
        derivatives.put( Activation.RELU, rel -> rel <= 0 ? 0.0 : 1.0);
        derivatives.put( Activation.SOFTMAX, s -> s * (1 -s));
    }

}
