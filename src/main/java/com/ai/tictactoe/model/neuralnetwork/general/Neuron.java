package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.jgrapht.Graphs;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
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

    /** Value of the neuron based on the aggregated sum of inputs multiplied by their weights **/
    Double netVal;

    /** Value of the neuron after applying activation function**/
    Double outputValue;

    /** Transient Value of calculated derivative of Error/output of this neuron -
     used during back propagation**/
    Double d_E_out;

    /** Transient Value of calculated derivative of Output/netVal of this neuron -
     used during back propagation**/
    Double d_out_net;

    /**
     * Default constructor
     * @param name - name of the neuron
     */
    public Neuron(final String name)
    {
        this.name = name;
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
    public List<DefaultWeightedEdge> getInputEdges(final SimpleDirectedWeightedGraph net)
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
    public List<DefaultWeightedEdge> getOutputEdges(final SimpleDirectedWeightedGraph net)
    {
        final List<DefaultWeightedEdge> outputEdges = new ArrayList<>();
        List<Neuron> successors = Graphs.successorListOf(net, this);
        successors.forEach( n -> {
            outputEdges.add((DefaultWeightedEdge)net.getEdge( this, n));
        });
        return outputEdges;
    }

    /**
     * Calculate sum of input weights from all predecessors multiplied by predecessor output values
     * @return aggregated net value of the neuron
     */
    public Double calcNetValueFromInputs(final SimpleDirectedWeightedGraph net)
    {
        List<Neuron> predecessors = Graphs.predecessorListOf(net, this);
        netVal = 0.00000d;
        predecessors.forEach( n -> {
            Double weightTimesInputVal = n.getOutputValue() * net.getEdgeWeight(net.getEdge(n, this));
            netVal += weightTimesInputVal;
        });
        return netVal;
    }

    /**
     * Calculates output value of the neuron by applying specific function.
     * @param activationFunction - function to apply eg. hyperbolic like SIGMOID or TANH
     * @return output value
     */
    public Double activate(Function<Double, Double> activationFunction)
    {
        outputValue = activationFunction.apply(netVal);
        return outputValue;
    }


}
