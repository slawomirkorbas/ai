package com.ai.tictactoe.model.neuralnetwork.general;

/**
 * Types of initialization of the edges (weights) within NeuralNetwork
 */
public enum WeightInitType
{
    /** No special weight initialization algorithm are used - just hardcoded values from layer configuration **/
    DEFAULT,

    /** Weights are initialized randomly between 0.1 and 0.9 **/
    RANDOM,

    /** Weights will be proportional to number of neurons within the layer **/
    XAVIER
}
