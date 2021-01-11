package com.ai.tictactoe.model.neuralnetwork.general;

import java.io.Serializable;

import java.util.function.BiFunction;

/**
 * Class aggregating various loss (cost) function types and their derivatives as lambda expressions
 */
public interface LossFunction extends BiFunction<Double, Double, Double>, Serializable
{
    /** Mean squared error **/
    LossFunction MSE = (y, t) -> 0.5 * Math.pow(t-y, 2);

    /** Cross Entropy - recommended when SOFTMAX transfer function is used **/ //https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
    LossFunction CROSS_ENTROPY = (t_true_distribution, y_estimated_distribution) -> {
        return (-1) * t_true_distribution * Math.log(y_estimated_distribution);
    };
}

