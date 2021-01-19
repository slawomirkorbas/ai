package com.ai.tictactoe.model.neuralnetwork.general

import org.jgrapht.graph.DefaultWeightedEdge
import org.jgrapht.graph.SimpleDirectedWeightedGraph
import spock.lang.Specification
import spock.lang.Unroll

class NeuralNetworkSpec extends Specification
{
    static final NeuralNetworkFactory nnf = new NeuralNetworkFactory()

    def "addLayer: works as expected"()
    {
        when:
            NeuralNetwork net = nnf.build()
               .input(27,  "P")
               .output(9, "O", 0.01d, TransferFunction.TANH,  CostFunction.MSE)
               .learningRate(0.01d)
               .initialize(WeightInitType.DEFAULT)

        then:
            net.getLayers().get(0).numberOfNeurons() == 27
            net.getLayers().get(1).numberOfNeurons() == 9
    }

    def "initialize: generate random weights properly"()
    {
        given:
            NeuralNetwork ann = nnf.build()
                    .input(5,  "P")
                    .hidden(4, "H", 0.01d, TransferFunction.SIGMOID)
                    .output(3, "O", 0.01d, TransferFunction.TANH,  CostFunction.MSE)
                    .initialize(WeightInitType.RANDOM)
        and:
            final SimpleDirectedWeightedGraph net = ann.getNet()

        expect:
             ann.getLayers().forEach(l -> {
                    l.getNeuronList().forEach( n ->
                    {
                        List<DefaultWeightedEdge> edges = n.getInputEdges(net)
                        edges.forEach( e -> {
                            assert(net.getEdgeWeight(e) <= 1.0d)
                            assert(net.getEdgeWeight(e) >= -1.0d)
                            System.out.println("Edge weight: " + net.getEdgeWeight(e))
                        })
                    })
                })
    }

    def "visualize: creates PNG file with neural network graph"()
    {
        given:
            NeuralNetwork net = nnf.build()
               .input(4, "P")
               .output(2, "H", 0.01d, TransferFunction.TANH,  CostFunction.MSE)
               .learningRate(0.01d)
               .initialize(WeightInitType.DEFAULT)

        when:
            File netImageFile = net.visualize()

        then:
            netImageFile != null
    }

    def 'predict: returns vector of expected size'()
    {
        given:
            NeuralNetwork net = nnf.build()
               .input(5, "I")
               .hidden(3, "H", 0.01d, TransferFunction.SIGMOID)
               .output(2, "O", 0.01d, TransferFunction.TANH, CostFunction.MSE)
               .learningRate(0.01d)
               .initialize(WeightInitType.DEFAULT)
        and:
            List<Double> inputs = [ 1.0d, 0.0d, 1.0d, 0.0d, 1.0d]

        when:
            List<Double> outputValues = net.predict(inputs)
        then:
            outputValues.size() == 2
            outputValues.forEach( v -> {
                v != null
            })
    }


    @Unroll
    def 'train: cost function value is minimized after executing specified number of train iterations'()
    {
        given:
            NeuralNetwork net = nnf.build()
                .input(3, "I")
                .output(2, "O", 0.1d, activationFunction, CostFunction.MSE)
                .learningRate(0.2d)
                .initialize(WeightInitType.DEFAULT)
        and:
            List<Double> outputs = net.predict(inputs)
            List<Double> squaredErrors = []
            outputs.stream().eachWithIndex{ double entry, int i ->
                squaredErrors.add( 0.5 * Math.pow(targets[i] - entry, 2))
            }

        when:
            5.times {
                net.train(inputs, targets, 1) }
        and:
            List<Double> newOutputs = net.predict(inputs)
        then:
            List<Double> updatedSquaredErrors = []
            newOutputs.stream().eachWithIndex{ double entry, int i ->
                updatedSquaredErrors.add( 0.5 * Math.pow(targets[i] - entry, 2))
                // new outputs value should be a little bit closer to expected(target) values
                Math.pow(targets[i] - entry,2) < Math.pow(targets[i] - outputs[i],2)
            }
        and:
            updatedSquaredErrors.stream().eachWithIndex { double err, int i ->
                err < squaredErrors[i]
            }

        where:
            activationFunction          | inputs               | targets
            TransferFunction.SIGMOID    | [ 1.0d, 0.0d, 1.0d ] | [ 1.0d, 0.0d ]
            TransferFunction.TANH       | [ 1.0d, 0.0d, 1.0d ] | [ 1.0d, 0.0d ]
            TransferFunction.SOFTMAX    | [ 1.0d, 0.0d, 1.0d ] | [ 1.0d, 0.0d ]
            TransferFunction.RELU       | [ 1.0d, 0.0d, 1.0d ] | [ 1.0d, 0.0d ]
    }


    def 'train test 1: 3 layer network can be trained to change binary input to decimal number in specific range'()
    {
        given:
            NeuralNetwork net = nnf.build()
                .input(2, "I")
                .hidden(3, "H1", 0.2d, TransferFunction.TANH)
                .hidden(3, "H2", 0.1d, TransferFunction.TANH)
                .output(1, "O", 0.05d, TransferFunction.RELU,  CostFunction.MSE)
                .initialize(WeightInitType.XAVIER) //TODO check why random doesn't work here sometimes bu XAVIER does...
        and:
            DataSet dataSet = new DataSet()
            dataSet.addExample([ 0.0d, 0.0d ], [0.0d])
            dataSet.addExample([ 0.0d, 1.0d ], [1.0d])
            dataSet.addExample([ 1.0d, 0.0d ], [2.0d])
            dataSet.addExample([ 1.0d, 1.0d ], [3.0d])

        when:
            net.train(dataSet)

        then:
            dataSet.examples.forEach( d -> {
                List<Double> predictedValues = net.predict(d.inputs)
                Integer prediction = Math.round(predictedValues[0])
                System.out.println("expected: " + d.targets[0] + ". predicted: " + prediction)
                assert (d.targets[0] == prediction)
            })
    }

    def 'train test 2: 4 layer network can be trained to change binary input to 1 if number is odd and to 0 if even'()
    {
        given:
            NeuralNetwork net = nnf.build()
                .input(2, "I")
                .hidden(4, "H1", 0.3d, TransferFunction.TANH)
                .hidden(3, "H2", 0.2d, TransferFunction.TANH)
                .output(1, "O", 0.1d, TransferFunction.TANH, CostFunction.MSE)
                .initialize(WeightInitType.RANDOM)
        and:
            DataSet dataSet = new DataSet()
            dataSet.addExample([ 0.0d, 0.0d ], [0.0d])
            dataSet.addExample([ 0.0d, 1.0d ], [1.0d])
            dataSet.addExample([ 1.0d, 0.0d ], [0.0d])
            dataSet.addExample([ 1.0d, 1.0d ], [1.0d])

        when:
            net.train(dataSet)

        then:
            dataSet.examples.forEach( e -> {
                List<Double> predictedValues = net.predict(e.inputs)
                System.out.println("expected: " + e.targets[0] + ". predicted: " + predictedValues[0])
                assert(e.targets[0] ==  Math.round(predictedValues[0]))
            })
    }


    def 'Network with SOFTMAX output can perform simple classification of ODD and EVEN numbers'()
    {
        given:
            NeuralNetwork net = nnf.build()
               .input(3, "I")
               .hidden(5, "H1", 0.3d, TransferFunction.TANH)
               .hidden(3, "H2", 0.2d, TransferFunction.TANH)
               .output(2, "O" , 0.1d, TransferFunction.SOFTMAX, CostFunction.CROSS_ENTROPY)
               .initialize(WeightInitType.RANDOM)
        and:
            DataSet dataSet = new DataSet()         // even, odd
            dataSet.addExample( [ 0.0d, 0.0d, 1.0d ], [0.0d, 1.0d])
            dataSet.addExample( [ 0.0d, 1.0d, 0.0d ], [1.0d, 0.0d])
            dataSet.addExample( [ 0.0d, 1.0d, 1.0d ], [0.0d, 1.0d])
            dataSet.addExample( [ 1.0d, 0.0d, 0.0d ], [1.0d, 0.0d])
            dataSet.addExample( [ 1.0d, 0.0d, 1.0d ], [0.0d, 1.0d])
            dataSet.addExample( [ 1.0d, 1.0d, 1.0d ], [1.0d, 0.0d])

        when:
            net.train(dataSet)
        then:
            dataSet.examples.forEach( d -> {
                    List<Double> predictedValues = net.predict(d.inputs)
                    System.out.println("target: " + d.targets[0] + ", " + d.targets[1] + ". predicted: " + predictedValues[0] + ", " + predictedValues[1])
                    assert(d.targets[0] == Math.round(predictedValues[0]))
        })
    }

    @Unroll
    def 'Softmax activation returns output probability values in range between 0 and 1 and they are sum upt to 1'()
    {
        given:
            NeuralNetwork ann = nnf.build()
                .input(4, "I")
                .output(3, "O" , 100d, TransferFunction.SOFTMAX, CostFunction.CROSS_ENTROPY)
                .learningRate(0.5d)
                .initialize(WeightInitType.DEFAULT)
        and:

        when:
            ann.predict(input)

        then:
            Double totalSum = 0.0
            ann.outputLayer.neuronList.forEach( v -> {
                v.outputValue >= 0.0d && v.outputValue <= 1.0d
                totalSum += v.outputValue
            })
        and:
            totalSum == 1.0d

        where:
            input << [[0.0d, 3.0d, 55.0d, -12.0d], [1000.0d, 3.0d, 2233.0d, -1222.0d]]
    }

    def 'serialization and deserialization works as expected'()
    {
        given:
            NeuralNetwork ann = nnf.build()
               .input(5, "I")
               .hidden(3, "H", 0.2d, TransferFunction.SIGMOID)
               .output(1, "O", 1.0d, TransferFunction.TANH, CostFunction.MSE)
               .learningRate(0.2d)
               .initialize(WeightInitType.RANDOM)
        and:
            final String annFileName = ann.serializeToFile(1,1)

        when:
            final NeuralNetwork readAnn = NeuralNetwork.deserialize(annFileName)
        then:
            readAnn.equals(ann)
            (new File(annFileName)).delete() // delete created file at the end
    }


//    def 'deepLearning4J: smoke test'()
//    {
//        given:
//            MultiLayerConfiguration configuration
//                    = new NeuralNetConfiguration.Builder()
//                    .activation(Activation.TANH)
//                    .weightInit(WeightInit.XAVIER)
//            //.regularization(true).l2(0.0001d)
//                    .list()
//                    .layer(0, new DenseLayer.Builder().nIn(2).nOut(8).build())
//                    .layer(1, new DenseLayer.Builder().nIn(8).nOut(8).build())
//                    .layer(2, new OutputLayer.Builder(
//                            LossFunctions.CostFunction.SQUARED_LOSS)
//                            .activation(Activation.RELU) //SOFTMAX
//                            .nIn(8).nOut(1).build())
//                    .build()
//        and:
//            MultiLayerNetwork net = new MultiLayerNetwork(configuration)
//            net.init()
//        and:
//            List dataSet = [ [ inputs: [ 0, 0 ], targets: [0.0d] ],
//                             [ inputs: [ 0, 1 ], targets: [1.0d] ],
//                             [ inputs: [ 1, 0 ], targets: [2.0d] ],
//                             [ inputs: [ 1, 1 ], targets: [3.0d] ] ]
//
//
//        when:
//            100.times {
//                dataSet.forEach( d -> {
//                    net.fit( Nd4j.create(d.inputs), Nd4j.create(d.targets) )
//                })
//            }
//        then:
//            dataSet.forEach( d -> {
//                int[] predictedValues = net.predict(Nd4j.create(d.inputs))
//                System.out.println("expected: " + d.targets[0] + ". predicted: " + predictedValues[0])
//                predictedValues[0] == d.targets[0]
//            })
//
//    }
}
