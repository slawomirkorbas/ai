package com.ai.tictactoe.model.neuralnetwork.general;

/**
 * Types of initialization of the edges (weights) within NeuralNetwork
 */
public enum WeightInitType
{
    /** No special weight initialization algorithm are used - just hardcoded values from layer configuration
     * not recommended as eg. similar or the same weight creates symtry within the network and the learning process
     * may be not effective **/
    DEFAULT,

    /** Weights are initialized randomly between -1.0 and 1.0 **/
    RANDOM,

    /** Weights will be proportional to number of neurons within the layer **/
    XAVIER
}
