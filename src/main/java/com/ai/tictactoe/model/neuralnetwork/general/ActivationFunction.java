package com.ai.tictactoe.model.neuralnetwork.general;

import java.io.Serializable;
import java.util.function.Function;

/**
 * Popular hyperbolic and derivatives of hyperbolic functions used in neural networks
 * as activation functions.
 */
public interface ActivationFunction extends  Function<Double, Double>, Serializable
{
    ActivationFunction RELU  = el -> Math.max(0, el);
    ActivationFunction SIGMOID  = el -> 1 / (1 + Math.exp(-el));
    ActivationFunction TANH     = el -> Math.sinh(el)/Math.cosh(el);
}
