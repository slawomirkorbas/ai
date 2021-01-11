package com.ai.tictactoe.model.neuralnetwork.general;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * Popular hyperbolic and derivatives of hyperbolic functions used in neural networks
 * as activation functions.
 */
public interface ActivationFunction extends Function<Double, Double>, Serializable
{

}
