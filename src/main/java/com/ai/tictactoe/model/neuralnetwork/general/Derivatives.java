package com.ai.tictactoe.model.neuralnetwork.general;

import java.util.function.Function;

/**
 * Simplified derivatives of some popular hyperbolic functions defined as functions
 * where the argument is interested hyperbolic function eg. TANH or SIGMOID.
 */
public class Derivatives
{
    public static final Function<Double, Double> DSIGMOID = sig -> sig * (1 - sig);
    public static final Function<Double, Double> DTANH    = tanh -> 1 - Math.pow(tanh,2);

    /**
     * REturn proper derivative function for given hyperbolic function
     * @param hyperbolicFun
     * @return
     */
    public static Function<Double, Double> getDerivative(Function<Double, Double> hyperbolicFun)
    {
        if(hyperbolicFun == ActivationFunction.SIGMOID)
        {
            return DSIGMOID;
        }
        if(hyperbolicFun == ActivationFunction.TANH)
        {
            return DTANH;
        }
        return null;
    }
}
