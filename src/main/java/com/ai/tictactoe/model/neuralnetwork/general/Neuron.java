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
    ActivationFunction activationFunction;

    /** Value of the neuron based on the aggregated sum of inputs multiplied by their weights **/
    Double netVal;

    /** Value of the neuron after applying activation function**/
    Double outputValue;

    /** Transient Value of calculated derivative of Error/netVal for this neuron - used during back propagation**/
    Double errorDeltaNet;

    /**
     * Default constructor
     * @param name - name of the neuron
     */
    public Neuron(final String name, ActivationFunction activationFunction)
    {
        this.name = name;
        this.activationFunction = activationFunction;
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
        if(activationFunction != null)
        {
            outputValue = activationFunction.apply(netVal);
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
     *
     * @param d_E_out
     * @return
     */
    public Double calculateErrorDeltaNet(Double d_E_out)
    {
        Double d_out_net = Activation.derivatives.get(activationFunction).apply(outputValue);
        errorDeltaNet = d_out_net * d_E_out;
        return d_out_net;
    }

    /**
     * Map of average gradients (Total error delta with respect specific input weight).
     * Calculated (averaged) fro each input weight during training iteration of the ANN.
     */
    final Map<DefaultWeightedEdge, Double> avgGradientPerInputWeightMap = new HashMap<>();

    /**
     * Not used so far...
     * @param inputEdge
     * @param sampleNo
     * @param d_Etotal_w
     * @return
     */
    public Double updateAverageGradientForWeight(final DefaultWeightedEdge inputEdge,
                                                 final int sampleNo,
                                                 final Double d_Etotal_w)
    {
        Double avg_gradient_per_weight = avgGradientPerInputWeightMap.get(inputEdge);
        if(avg_gradient_per_weight == null)
        {
            avg_gradient_per_weight = 0.0;
        }
        avg_gradient_per_weight = ((avg_gradient_per_weight + d_Etotal_w)/sampleNo);
        avgGradientPerInputWeightMap.put(inputEdge, avg_gradient_per_weight);
        return avg_gradient_per_weight;
    }




}
