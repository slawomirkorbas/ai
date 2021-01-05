package com.ai.tictactoe.model.neuralnetwork.general;

import java.io.Serializable;
import java.util.function.Function;

/**
 * Popular hyperbolic and derivatives of hyperbolic functions used in neural networks
 * as activation functions.
 */
public interface ActivationFunction extends  Function<Double, Double>, Serializable
{
    ActivationFunction RELU     = net -> Math.max(0, net);
    ActivationFunction SIGMOID  = net -> 1 / (1 + Math.exp(-net));
    ActivationFunction TANH     = net -> Math.sinh(net)/Math.cosh(net);
}
