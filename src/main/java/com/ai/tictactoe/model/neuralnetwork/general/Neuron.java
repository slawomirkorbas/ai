package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.jgrapht.Graphs;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Class representing single neuron in ANN graph.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class Neuron implements Serializable
{
    private static final long serialVersionUID = 1L;

    /** Name of the neuron **/
    String name;

    /** Activation function to apply on Net value eg. hyperbolic like SIGMOID or TANH **/
    TransferFunction transferFunction;

    /** Value of the neuron based on the aggregated sum of inputs multiplied by their weights **/
    Double netVal;

    /** Value of the neuron after applying activation function**/
    Double outputValue;

    /** Transient Value of calculated derivative of Error/netVal for this neuron - used during back propagation**/
    Double errorDeltaNet;

    /** Derivative of total error with respect to output value of this neuron - transient value used during backprop **/
    Double d_E_total_out;

    /** Flag indicating whether this is a bias neuron with fixed value **/
    boolean bias = false;

    /** The layer to which this neuron belongs to **/
    Layer parentLayer;

    /**
     * Default constructor
     * @param name - name of the neuron
     */
    public Neuron(final String name, TransferFunction transferFunction, Layer parent)
    {
        this.name = name;
        this.transferFunction = transferFunction;
        this.parentLayer = parent;
    }

    /**
     * Optional constructor
     * @param name
     * @param transferFunction
     */
    public Neuron(final String name, TransferFunction transferFunction)
    {
        this.name = name;
        this.transferFunction = transferFunction;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null) return false;
        if (this.getClass() != o.getClass()) return false;

        return this.name.equals(((Neuron)o).name);
    }

    @Override
    public int hashCode()
    {
        return this.name.hashCode();
    }

    /**
     * Returns true if neuron belongs to input layer
     * @return
     */
    public boolean isInput()
    {
        return this.parentLayer.isInputLayer();
    }

    /**
     * Returns list of input edges leading to this neuron
     * @param net
     * @return
     */
    public List<DefaultWeightedEdge> getInputEdges(final SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> net)
    {
        final List<DefaultWeightedEdge> inputEdges = new ArrayList<>();
        List<Neuron> predecessors = Graphs.predecessorListOf(net, this);
        predecessors.forEach( n -> {
            inputEdges.add((DefaultWeightedEdge)net.getEdge(n, this));
        });
        return inputEdges;
    }

    /**
     * Returns list of output edges from this neuron
     * @param net
     * @return
     */
    public List<DefaultWeightedEdge> getOutputEdges(final SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> net)
    {
        final List<DefaultWeightedEdge> outputEdges = new ArrayList<>();
        List<Neuron> successors = Graphs.successorListOf(net, this);
        successors.forEach( n -> {
            outputEdges.add(net.getEdge( this, n));
        });
        return outputEdges;
    }

    /**
     * Calculate sum of input weights from all predecessors multiplied by predecessor output values
     * @return aggregated net value of the neuron
     */
    public Double calcNetValue(final SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> net)
    {
        List<Neuron> predecessors = Graphs.predecessorListOf(net, this);
        netVal = 0.0;
        predecessors.forEach( n -> {
            Double weightTimesInputVal = n.getOutputValue() * net.getEdgeWeight(net.getEdge(n, this));
            netVal += weightTimesInputVal;
        });
        return netVal;
    }

    /**
     * Calculates output value of the neuron by applying specific function.
     * @return output value
     */
    public Double activate()
    {
        if(transferFunction != null)
        {
            outputValue = Activation.activations.get(transferFunction).apply(netVal);
        }
        return outputValue;
    }

    /**
     * Activates (calculate an output value of the neuron) using SOFTMAX function which requires
     * total sum of exponents of net values from other neurons in this layer.
     * @param netValues - net values from all neurons within the layer
     *
     * @return output value
     */
    public Double activateWithSoftmax(final List<Double> netValues)
    {
        Double maxNetVal = Collections.max(netValues);
        Double totalExpSum = netValues.stream()
                                      .map(netVal -> Math.exp(netVal - maxNetVal))
                                      .reduce(0.0, Double::sum);
        outputValue = (Math.exp(netVal - maxNetVal)/totalExpSum);
        return outputValue;
    }


    /**
     * Map of average gradients (Total error delta with respect specific input weight).
     * Calculated (averaged) fro each input weight during training iteration of the ANN.
     */
    final Map<DefaultWeightedEdge, Double> avgGradientPerInputWeightMap = new HashMap<>();

}
