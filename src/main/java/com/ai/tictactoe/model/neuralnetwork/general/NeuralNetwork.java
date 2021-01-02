package com.ai.tictactoe.model.neuralnetwork.general;

import com.mxgraph.layout.mxCompactTreeLayout;
import com.mxgraph.layout.mxIGraphLayout;
import com.mxgraph.util.mxCellRenderer;
import lombok.Getter;
import lombok.Setter;
import org.jgrapht.Graphs;
import org.jgrapht.ext.JGraphXAdapter;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;

import javax.imageio.ImageIO;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class representing neural network (ANN)
 */
public class NeuralNetwork implements Serializable
{
    private static final long serialVersionUID = 1L;

    /** Weighted graph representing neural network **/
    private SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> net;

    /** Default learning rate **/
    private double learningRate = 0.01;

    /** List of hidden layers**/
    @Getter
    @Setter
    private List<Layer> layers;

    /**
     * No args constructor.
     */
    public NeuralNetwork()
    {

    }

    /**
     * Default constructor
     */
    public NeuralNetwork(double learningRate)
    {
        this.net = new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
    }

    /**
     * Adds new layer to the network.
     * @param noOfNeurons - number of neurons within the layer
     * @param initWeight - initial weight for edges between the new layer and previous layer. MAy be null if this is the first layer.
     * @param layerName - layer name (should be unique)
     * @param activationFunction - hyperbolic activation function to be applied for neurons during learning or evaluation process.
     * @return newly created layer
     */
    public Layer addLayer(final int noOfNeurons, final String layerName, final Double initWeight, ActivationFunction activationFunction)
    {
        if(net.vertexSet().stream().anyMatch( n -> n.name.contains("_" + layerName + "_")))
        {
            System.out.println("Layer with this name already exists.");
            return null;
        }

        Layer newLayer;
        final int numberOfLayers = layers.size();

        // find all neurons(vertexes) from previous layer
        final Layer previousLayer = numberOfLayers > 0 ? layers.get(numberOfLayers- 1) : null;

        for(int neuronIndex = 0; neuronIndex < noOfNeurons; neuronIndex++)
        {
            // add new neuron
            Neuron newNeuron = new Neuron(numberOfLayers + "_" + layerName + "_" + neuronIndex);
            net.addVertex(newNeuron);

            // add weighted connections to previous layer
            if(previousLayer != null)
            {
                for(Neuron neuron : previousLayer.getNeuronList())
                {
                    DefaultWeightedEdge newEdge = net.addEdge(neuron, newNeuron);
                    net.setEdgeWeight(newEdge, initWeight);
                }
            }
        }

        newLayer = new Layer(layerName, net.vertexSet().stream()
                              .filter(n -> n.name.startsWith(numberOfLayers + "_")).collect(Collectors.toList()),
                              activationFunction
                             );

        // adds layer to the main list
        layers.add(newLayer);

        return newLayer;
    }


    /**
     * Do prediction of output vaues based on specific weights and biases of neural network
     * @param inputValues - input values to feed neurons from the input layer
     * @return list of values from output neurons
     */
    public List<Double> predict(List<Integer> inputValues)
    {
        if(layers.size() == 0)
        {
            return null;
        }
        // init value for each neuron in the input layer
        Layer inputLayer = layers.get(0);
        for(int i = 0; i < inputLayer.numberOfNeurons(); i++)
        {
            inputLayer.get(i).setOutputValue(Double.valueOf(inputValues.get(i)));
        }

        // calculate net value and activate each neuron inside every hidden layer
        for( int i=1; i < layers.size(); i++)
        {
            layers.get(i).doForwardPass(net);
        }

        // return neurons from the last layer (output layer)
        return layers.get(layers.size()-1).neuronList.stream().map(n -> n.getOutputValue()).collect(Collectors.toList());
    }


    /**
     *
     * @param inputs
     * @param targets
     */
    public void train(final List<Integer> inputs,  final List<Double> targets)
    {
        if(layers.size() < 2 || inputs.isEmpty() || targets.isEmpty())
        {
            return;
        }

        // predict results for given inputs...
        predict(inputs);

        // Back propagation:
        // Calculate derivative of the cost(error) function with respect to input weights of specific layer

        // ... for the output layer and all preceding hidden layers
        for(int l = layers.size() - 1; l > 0; l-- )
        {
            final Layer currentLayer = layers.get(l);
            //for each neuron in the layer:
            for(int i = 0; i < currentLayer.numberOfNeurons(); i++)
            {
                final Neuron currentNeuron = currentLayer.get(i);
                Double d_outi_neti = null;
                Double d_Etotal_outi = null;
                if(currentNeuron.getOutputEdges(net).size() == 0 )
                {
                    // Output Layer:
                    // Partial derivative of E (cost function value) with respect to Out (activation result(output))
                    // derivative of squared error: 0.5 * Math.pow(targets.get(i) - currentNeuron.getOutputValue(),2)
                    d_Etotal_outi = -(targets.get(i) - currentNeuron.getOutputValue());

                    // Partial derivatives of Output(activation function results) with respect to Net value of the neuron
                    d_outi_neti = Derivatives.getDerivative(currentLayer.getActivationFunction()).apply(currentNeuron.getOutputValue());

                    // save calculated partial derivatives for this neuron
                    currentNeuron.d_out_net = d_outi_neti;
                    currentNeuron.d_E_out = d_Etotal_outi;
                }
                else
                {
                    // Hidden layer:
                    // Partial derivatives of Output(activation function results) with respect to Net value of the neuron
                    d_outi_neti = Derivatives.getDerivative(currentLayer.getActivationFunction()).apply(currentNeuron.getOutputValue());
                    List<DefaultWeightedEdge> outputEdges = currentNeuron.getOutputEdges(net);
                    for(DefaultWeightedEdge edge : outputEdges)
                    {
                        Double d_Ei_netoi = net.getEdgeTarget(edge).d_E_out * net.getEdgeTarget(edge).d_out_net;
                        Double d_netoi_outhi = net.getEdgeWeight(edge);  // this is just "wi" (Weight) because: (wi*outhi + wj*outhj)' = wi
                        Double d_Ei_outhi = d_Ei_netoi * d_netoi_outhi;
                        if(d_Etotal_outi == null) {
                            d_Etotal_outi = d_Ei_outhi;
                        }
                        else {
                            d_Etotal_outi += d_Ei_outhi;
                        }
                    }

                    // save calculated partial derivatives for this neuron
                    currentNeuron.d_out_net = d_outi_neti;
                    currentNeuron.d_E_out = d_Etotal_outi;
                }

                // Partial derivative of Net with respect to specific input weight (i,j)
                // this is just Output value of the predecessor as (Net)' = (OUT_0 * W_01 + ... OUT_i * W_ij)' = OUT_0 (just constant)
                List<Neuron> predecessors = Graphs.predecessorListOf(net, currentNeuron);
                for( int p = 0; p < predecessors.size(); p++)
                {
                    final Neuron predecessor = predecessors.get(p);

                    //Apply chaining rule to calculate d_E_w
                    Double d_net_w = predecessor.getOutputValue();
                    Double d_Etotal_w = d_net_w * d_outi_neti * d_Etotal_outi;

                    // subtract calculated delta multiplied by learning rate from the weight (i,j)
                    DefaultWeightedEdge edge = net.getEdge(predecessor, currentNeuron);
                    Double weight = net.getEdgeWeight(edge);
                    net.setEdgeWeight(edge, weight - learningRate * d_Etotal_w);
                }
            }
        }
    }

    /**
     * Render neural network to image file.
     * @return file containing network visualization
     * @throws IOException
     */
    public File visualize() throws IOException
    {
        JGraphXAdapter<Neuron, DefaultWeightedEdge> graphAdapter = new JGraphXAdapter<Neuron, DefaultWeightedEdge>(net);
        mxIGraphLayout layout = new mxCompactTreeLayout(graphAdapter);
        layout.execute(graphAdapter.getDefaultParent());

        BufferedImage image = mxCellRenderer.createBufferedImage(graphAdapter, null, 2, Color.WHITE, true, null);
        File imgFile = new File("src/test/resources/network.png");
        ImageIO.write(image, "PNG", imgFile);

        return imgFile;
    }


    /**
     * Serialize Neural Network
     * @return byte output stream
     */
    private ByteArrayOutputStream serialize() throws IOException
    {

        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
        objectOutputStream.writeObject(this);
        objectOutputStream.flush();
        objectOutputStream.close();
        return byteArrayOutputStream;
    }


    /**
     * Save Neural Network data to file
     * @return file name
     */
    public String serializeToFile()
    {
        try
        {
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd-HHmm");
            final String fileName = "neural-network-" + formatter.format(LocalDateTime.now()) + ".ann";
            ByteArrayOutputStream stream = serialize();
            FileOutputStream fileOutputStream = new FileOutputStream(fileName);
            fileOutputStream.write(stream.toByteArray());
            fileOutputStream.close();
            return fileName;
        }
        catch(IOException e)
        {
            return null;
        }
    }

    /**
     * Reads NeuralNetwork object from file
     * @param fileName <code>*.ann</code> file storing artificial neural network
     * @return NeuralNetwork object
     */
    public static NeuralNetwork deserialize(final String fileName)
    {
        try
        {
            final File file = new File(fileName);
            final FileInputStream fileInputStream = new FileInputStream(file);
            final ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
            NeuralNetwork neuralNetwork = (NeuralNetwork) objectInputStream.readObject();
            fileInputStream.close();
            return neuralNetwork;
        }
        catch(IOException | ClassNotFoundException e)
        {
            return null;
        }
    }

    @Override
    public int hashCode()
    {
        return this.net.hashCode();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null) return false;
        if (this.getClass() != o.getClass()) return false;

        //compare two objects layer by layer...
        NeuralNetwork netA = (NeuralNetwork)o;
        List<Layer> layers = netA.getLayers();
        for(int l=0; l < layers.size(); l++)
        {
            final Layer layer = layers.get(l);
            if(!layer.equals(this.layers.get(l)))
            {
                return false;
            }

            //compare Edges fo neurons
            // TODO remove this code if native jgrapht "equals" will start to work...
            List<Neuron> neuronListA = layer.getNeuronList();
            List<Neuron> neuronList = this.layers.get(l).getNeuronList();
            for(int i=0; i<neuronListA.size(); i++)
            {
                Neuron nA = neuronListA.get(i);
                Neuron nB = neuronList.get(i);

                List<DefaultWeightedEdge> inputEdgesA = nA.getInputEdges(netA.net);
                List<DefaultWeightedEdge> inputEdges = nB.getInputEdges(this.net);
                for(int e=0; e < inputEdgesA.size(); e++)
                {
                    if(netA.net.getEdgeWeight(inputEdgesA.get(e)) != this.net.getEdgeWeight(inputEdges.get(e))) {
                        return false;
                    }
                }
            }
        }

        //finally compare internal graph topology
        //TODO native jgrapht "equals" fails and this should be investigated...
        //return this.net.equals(netA.net);
        return true;
    }

}
