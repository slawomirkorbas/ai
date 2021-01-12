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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
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
    private double learningRate = DEFAULT_LEARNING_RATE;

    /** Algorithm identifier which is used to initialize weights between layer **/
    private WeightInitializationType weightInitializationType;

    /** Weight initializer **/
    private static final Double DEFAULT_LEARNING_RATE = 0.01d;

    /** Derivatives of specific loss functions **/
    public static final Map<LossFunction, BiFunction<Double, Double, Double>> lossDerivatives = new HashMap<>();
    static
    {
        lossDerivatives.put( LossFunction.MSE, (t, y) -> -(t - y));
        lossDerivatives.put( LossFunction.CROSS_ENTROPY, (t, y) -> y - t);
    }

    /** List of layers**/
    @Getter
    @Setter
    private List<Layer> layers;

    /**
     * No args constructor.
     */
    public NeuralNetwork()
    {
        this.net = new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        this.layers = new ArrayList<>();
        this.learningRate = DEFAULT_LEARNING_RATE;
        this.weightInitializationType = WeightInitializationType.NONE;
    }

    /**
     * Default constructor
     *
     * @param learningRate
     * @param weightInitializationType
     */
    public NeuralNetwork(double learningRate, WeightInitializationType weightInitializationType)
    {
        this.net = new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.weightInitializationType = weightInitializationType;
    }

    /**
     * Returns an output layer
     * @return
     */
    public Layer getOutputLayer()
    {
        return layers.get(layers.size()-1);
    }

    /**
     * Returns an input layer
     * @return
     */
    public Layer getInputLayer()
    {
        return layers.get(0);
    }

    /**
     * True if network has just one output neuron.
     * @return
     */
    public boolean isSingleOutput()
    {
        return getOutputLayer().getNeuronList().size() == 1;
    }

    /**
     * Adds the new layer to the network
     * @param layer
     * @return
     */
    public NeuralNetwork layer(final Layer layer)
    {
        if(layerExists(layer.getName()))
        {
            System.out.println("Layer with this name already exists.");
            return null;
        }

        final int numberOfLayers = layers.size();
        final Layer previousLayer = numberOfLayers > 0 ? layers.get(numberOfLayers-1) : null;
        layer.connectToPreviousLayer(net, previousLayer);

        //create "Bias" neuron: biases are needed because if an input pattern are zero, the weights
        //may not be never changed for this pattern and the net could not learn it. This is kind of
        //"pseudo input" with constant value of "1"
        if(previousLayer != null) // For all layers except input layer
        {
            layer.addBiases(net);
        }
        layers.add(layer);
        return this;
    }

    private boolean layerExists(String name)
    {
        return net.vertexSet().stream().anyMatch( n -> n.name.contains("_" + name + "_"));
    }

    /**
     * Initialize weights using specific Weight initialization algorithm
     */
    public NeuralNetwork initialize()
    {
        switch(this.weightInitializationType)
        {
            case NONE: break;
            case XAVIER: // not sure if it is real XAVIER...
                layers.stream().forEach( l -> {
                    l.getNeuronList().stream().forEach( n -> {
                        List<DefaultWeightedEdge> inputEdges = n.getInputEdges(net);
                        if(inputEdges.size() > 0)
                        {
                            Double xavierWeight = Double.valueOf(1.0/inputEdges.size());
                            for( DefaultWeightedEdge edge : inputEdges)
                            {
                                net.setEdgeWeight(edge, xavierWeight);
                            }
                        }
                    });
                });
                break;
        }
        return this;
    }


    /**
     * Do prediction of output vaues based on specific weights and biases of neural network
     * @param inputValues - input values to feed neurons from the input layer
     * @return list of values from output neurons
     */
    public List<Double> predict(final List<Integer> inputValues)
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
            layers.get(i).forwardPass(net);
        }

        // return neurons from the last layer (output layer)
        return layers.get(layers.size()-1).neuronList.stream().map(n -> n.outputValue).collect(Collectors.toList());
    }

    /**
     * Train neural network using supervised learning (example data targets).
     * Uses "back propagation" algorithm.
     *
     * @param inputs
     * @param targets
     * @param sampleNumber
     */
    public void train(final List<Integer> inputs,  final List<Double> targets, int sampleNumber)
    {
        if(layers.size() < 2 || inputs.isEmpty() || targets.isEmpty())
        {
            return;
        }

        // predict results for given inputs...
        predict(inputs);

        // Back propagation: calculate derivative of the cost(error) function with respect to input weights of specific layer's neurons
        for(int l = layers.size() - 1; l > 0; l-- )
        {
            final Layer currentLayer = layers.get(l);
            for(int i = 0; i < currentLayer.numberOfNeurons(); i++)
            {
                final Neuron currentNeuron = currentLayer.get(i);
                Double d_E_out = 0.0;
                if(currentLayer.isOutputLayer())
                {
                    // Partial derivative of E (cost function value) with respect to Out (activation result(output))
                    d_E_out = lossDerivatives.get(((OutputLayer)currentLayer).lossFunction).apply(targets.get(i), currentNeuron.outputValue);
                }
                else // Hidden layer:
                {
                    // Partial derivatives of Output(activation function results) with respect to Net value of the neuron
                    List<DefaultWeightedEdge> outputEdges = currentNeuron.getOutputEdges(net);
                    for(DefaultWeightedEdge edge : outputEdges)
                    {
                        Double d_Ei_netoi = net.getEdgeTarget(edge).errorDeltaNet;
                        Double d_Netoi_outhi = net.getEdgeWeight(edge);  // this is just "wi" (Weight) because: (wi*outhi + wj*outhj)' = wi
                        Double d_Ei_outhi = d_Ei_netoi * d_Netoi_outhi;
                        d_E_out += d_Ei_outhi;
                    }
                }
                currentNeuron.calculateErrorDeltaNet(d_E_out);

                // Partial derivative of Net with respect to specific input weight (i,j)
                List<Neuron> predecessors = Graphs.predecessorListOf(net, currentNeuron);
                //System.out.println("Updating weights: ");
                for(int p = 0; p < predecessors.size(); p++)
                {
                    final Neuron predecessor = predecessors.get(p);
                    //Apply chaining rule to calculate d_E_w
                    Double d_net_w = predecessor.outputValue;
                    Double d_Etotal_w = d_net_w * currentNeuron.errorDeltaNet;
                    DefaultWeightedEdge edge = net.getEdge(predecessor, currentNeuron);
                    net.setEdgeWeight(edge, net.getEdgeWeight(edge) - learningRate * d_Etotal_w);
                    //System.out.println("w(" + predecessor.getName() + "->" + currentNeuron.getName() + ") updated: " + net.getEdgeWeight(edge));
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
            String fileName = "net";
            for(Layer l : this.layers)
            {
                fileName += "-" + l.getNeuronList().size();
            }
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd-HHmm");
            fileName += "-" + formatter.format(LocalDateTime.now()) + ".ann";
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
