package com.ai.tictactoe.model.neuralnetwork.general;

/**
 * Enum aggregating various loss (cost)
 */
public enum CostFunction
{
    /** Mean squared error **/
     MSE,

    /** Cross Entropy - recommended when SOFTMAX transfer function is used **/ //https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
     CROSS_ENTROPY
}

