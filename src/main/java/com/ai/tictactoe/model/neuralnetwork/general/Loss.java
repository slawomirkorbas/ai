package com.ai.tictactoe.model.neuralnetwork.general;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

public class Loss
{
    public static final Map<CostFunction, BiFunction<Double, Double, Double>> functions = new HashMap<>();
    static
    {
        functions.put(CostFunction.MSE, (y, t) -> 0.5 * Math.pow(t-y, 2));
        functions.put(CostFunction.CROSS_ENTROPY, (t_true_distribution, y_estimated_distribution) -> (-1) * t_true_distribution * Math.log(y_estimated_distribution));
    }

    /** Derivatives of specific loss functions **/
    public static final Map<CostFunction, BiFunction<Double, Double, Double>> derivatives = new HashMap<>();
    static
    {
        derivatives.put(CostFunction.MSE, (t, y) -> -(t - y));
        derivatives.put(CostFunction.CROSS_ENTROPY, (t, y) -> y - t);
    }
}
