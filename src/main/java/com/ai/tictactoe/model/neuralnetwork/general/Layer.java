package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.Data;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;
import java.io.Serializable;
import java.util.List;
import java.util.function.Function;

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

    /** The activation function used by the neuron within the layer **/
    ActivationFunction activationFunction;

    /**
     * Default constructor.
     * @param name
     * @param neuronList
     * @param function
     */
    public Layer(final String name, final List<Neuron> neuronList, ActivationFunction function)
    {
        this.name = name;
        this.activationFunction = function;
        this.neuronList = neuronList;
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
    public void doForwardPass(final SimpleDirectedWeightedGraph net)
    {
        for(Neuron neuron : this.neuronList)
        {
            neuron.calcNetValueFromInputs(net);
            neuron.activate(activationFunction);
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
        if(layer.activationFunction != null &&
            this.activationFunction == null) return false;
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
