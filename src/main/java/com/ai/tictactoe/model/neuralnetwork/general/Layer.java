package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.Data;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class representing specific layer of neurons within the Neural network
 */
@Data
public class Layer implements Serializable
{
    /** Unique layer name **/
    String name;

    /** List of neurons within the layer **/
    List<Neuron> neuronList;

    /** Initial weight for the neurons within layer **/
    Double initialWeight;

    /** The activation function used by the neuron within the layer **/
    TransferFunction transferFunction;


    boolean outputLayer = false;
    boolean inputLayer = false;

    /**
     * Default constructor
     * @param name
     * @param noOfNeurons
     * @param function
     */
    public Layer(final int noOfNeurons,
                 final String name,
                 final Double initialWeight,
                 TransferFunction function)
    {
        this.name = name;
        this.initialWeight = initialWeight;
        this.transferFunction = function;
        this.neuronList = new ArrayList<>();
        for(int neuronIndex = 0; neuronIndex < noOfNeurons; neuronIndex++)
        {
            // add new neuron
            neuronList.add(new Neuron(name + "_" + neuronIndex, transferFunction, this));
        }
    }

    /**
     * Connects all neurons of this layer to all neurons from previous layer (N:N)
     * @param net
     * @param previousLayer
     */
    public void connectToPreviousLayer(final SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> net,
                                       final Layer previousLayer)
    {
        for(Neuron n : neuronList)
        {
            net.addVertex(n);
            if(previousLayer != null)
            {
                for(Neuron neuron : previousLayer.getNeuronList())
                {
                    DefaultWeightedEdge newEdge = (DefaultWeightedEdge) net.addEdge(neuron, n);
                    net.setEdgeWeight(newEdge, initialWeight);
                }
            }
        }

    }

    /**
     * Adds biases to all neurons within the layer
     * @param net
     */
    public void addBias(final SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> net)
    {
        Neuron bias = new BiasNeuron("Bias_" + name + "_", 1.0);
        net.addVertex(bias);

        //connect the "Bias" neuron to each neuron from this layer
        for(Neuron n : neuronList)
        {
            DefaultWeightedEdge newEdge = net.addEdge(bias, n);
            net.setEdgeWeight(newEdge, initialWeight);
        }
    }

    /**
     * Return size of the layer (number of neurons)
     * @return
     */
    public int numberOfNeurons()
    {
        return this.neuronList.size();
    }

    /**
     * Return Neuron with given index
     * @param index
     * @return
     */
    public Neuron get(final int index)
    {
        return this.neuronList.get(index);
    }


    /**
     * Calculate net(Z value) and activate each neuron
     * @param net - network graph
     */
    public void forwardPass(final SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> net)
    {
        if(transferFunction.equals(TransferFunction.SOFTMAX))
        {
            neuronList.stream().forEach(n -> n.calcNetValue(net));
            List<Double> netValuesVector = neuronList.stream().map(n -> n.netVal).collect(Collectors.toList());
            neuronList.stream().forEach(n -> n.activateWithSoftmax(netValuesVector));
        }
        else
        {
            neuronList.stream().forEach(n -> {
                                            n.calcNetValue(net);
                                            n.activate();
                                        });
        }
    }


    @Override
    public int hashCode()
    {
        return this.name.hashCode();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null) return false;
        if (this.getClass() != o.getClass()) return false;

        // Compare layer properties
        final Layer layer = (Layer)o;
        if( !layer.name.equals(this.name)) return false;
        if(layer.transferFunction != null &&
            this.transferFunction == null) return false;
        if( layer.numberOfNeurons() != this.numberOfNeurons()) return false;

        //Compare neurons...
        for(int n=0; n < this.numberOfNeurons(); n++)
        {
            if(!this.get(n).equals(layer.neuronList.get(n)))
            {
                return false;
            }
        }
        return true;
    }

}
