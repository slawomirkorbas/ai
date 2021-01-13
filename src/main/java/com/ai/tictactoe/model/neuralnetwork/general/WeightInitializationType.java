package com.ai.tictactoe.model.neuralnetwork.general;

/**
 * Types of initialization of the edges (weights) within NeuralNetwork
 */
public enum WeightInitializationType
{
    /** No special weight initialization algorithm are used - just hardcoded values from layer configuration **/
    DEFAULT,

    /** Weights will be proportional to number of neurons within the layer **/
    XAVIER
}
