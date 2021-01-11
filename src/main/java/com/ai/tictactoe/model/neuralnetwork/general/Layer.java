package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.Data;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Class representing specific layer of neurons within the Neural network
 */
@Data
public class Layer implements Serializable
{
    /** Unique layer name **/
    String name;

    /** Number of layer **/
    Integer layerNo;

    /** List of neurons within the layer **/
    List<Neuron> neuronList;

    /** Initial weight for the neurons within layer **/
    Double initialWeight;

    /** The activation function used by the neuron within the layer **/
    ActivationFunction activationFunction;

    /** Normalizer constant for SOFTMAX activation function. This is used to decrease exponent value and avoid NaN outputs.
     * - see: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
     * **/
    Double softMaxNormalizerConstant;

    /**
     * Default constructor
     * @param name
     * @param noOfNeurons
     * @param function
     */
    public Layer(final int noOfNeurons,
                 final String name,
                 final Double initialWeight,
                 ActivationFunction function)
    {
        this.name = name;
        this.layerNo = 0;//layerNo;
        this.initialWeight = initialWeight;
        this.activationFunction = function;
        this.neuronList = new ArrayList<>();
        for(int neuronIndex = 0; neuronIndex < noOfNeurons; neuronIndex++)
        {
            // add new neuron
            neuronList.add(new Neuron(layerNo + "_" + name + "_" + neuronIndex, activationFunction));
        }
    }

    /**
     * Connects all neurons of this layer to all neurons from previous layer (N:N)
     * @param net
     * @param previousLayer
     */
    public void connectToPreviousLayer(final SimpleDirectedWeightedGraph net,
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
    public void addBiases(final SimpleDirectedWeightedGraph net)
    {
        Neuron bias = new Neuron(this.layerNo + "_Bias_" + name + "_", null);
        bias.setOutputValue(1.0);
        net.addVertex(bias);

        //connect the "Bias" neuron to each neuron from this layer
        for(Neuron n : neuronList)
        {
            DefaultWeightedEdge newEdge = (DefaultWeightedEdge)net.addEdge(bias, n);
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
    public void forwardPass(final SimpleDirectedWeightedGraph net)
    {
        if(activationFunction == Activation.SOFTMAX)
        {
            // Calculate net values and normalizing constant (maximum between all inputs, negated)
            softMaxNormalizerConstant = null;
            neuronList.stream().forEach(n -> {
                Double max = n.calcNetValueFromInputs(net);
                softMaxNormalizerConstant = softMaxNormalizerConstant == null ? max : Math.max(max, softMaxNormalizerConstant);
            });

            // calculate softmax denominator (sum of all exponents)
            final Double softmaxDenominator = neuronList.stream().map(n -> Math.exp(n.netVal - softMaxNormalizerConstant)).reduce(0.0, Double::sum);

            // calculate softmax value for each output
            for(Neuron neuron : this.neuronList)
            {
                neuron.activateWithSoftmax(softmaxDenominator, softMaxNormalizerConstant);
            }
        }
        else
        {
            neuronList.stream().forEach(n -> {
                                            n.calcNetValueFromInputs(net);
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

    boolean isOutputLayer()
    {
        return false;
    }
}
