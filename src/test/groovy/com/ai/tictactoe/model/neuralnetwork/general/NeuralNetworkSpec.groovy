package com.ai.tictactoe.model.neuralnetwork.general

import spock.lang.Specification
import spock.lang.Unroll

class NeuralNetworkSpec extends Specification
{
    def "addLayer: works as expected"()
    {
        given:
            double learningRate = 0.01
            NeuralNetwork net = new NeuralNetwork(learningRate, WeightInitializationType.NONE)

        when:
            Layer inputLayer = net.addLayer(27, "P", null, null)
            Layer hiddenLayer = net.addLayer(9, "H", 0.01d, Activation.TANH)

        then:
            inputLayer.numberOfNeurons() == 27
            hiddenLayer.numberOfNeurons() == 9
    }

    def "visualize: creates PNG file with neural network graph"()
    {
        given:
            double learningRate = 0.01
            NeuralNetwork net = new NeuralNetwork(learningRate, WeightInitializationType.NONE)
            net.addLayer(4, "P", null, null)
            net.addLayer(2, "H", 0.01d, Activation.TANH)

        when:
            File netImageFile = net.visualize()

        then:
            netImageFile != null
    }

    def 'predict: returns vector of expected size'()
    {
        given:
            NeuralNetwork net = new NeuralNetwork(0.01, WeightInitializationType.NONE)
            net.addLayer(5, "I", null, null)
            net.addLayer(3, "H", 0.01d, Activation.SIGMOID)
            net.addLayer(2, "O", 0.01d, Activation.TANH)
        and:
            List<Integer> inputs = [ 1, 0, 1, 0, 1]

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
            NeuralNetwork net = new NeuralNetwork(0.2d, WeightInitializationType.NONE)
            net.addLayer(3, "I", null, null)
            net.addLayer(2, "H", 0.1d, activationFunction)
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
            activationFunction | inputs      | targets
            Activation.SIGMOID | [ 1, 0, 1 ] | [ 1.0d, 0.0d ]
            Activation.TANH    | [ 1, 0, 1 ] | [ 1.0d, 0.0d ]
    }


    def 'train test 1: 3 layer network can be trained to change binary input to decimal number in specific range'()
    {
        given:
            final NeuralNetwork net = new NeuralNetwork(0.2d, WeightInitializationType.NONE)
            net.addLayer(2, "I", null, null)
            net.addLayer(3, "H1", 0.2d, Activation.TANH)
            net.addLayer(3, "H2", 0.1d, Activation.TANH)
            //net.addLayer(3, "H3", 0.05d, Activation.SIGMOID)
            net.addLayer(1, "O", 0.05d, Activation.RELU)
            net.initialize()

        when:
            List dataSet = [ [ inputs: [ 0, 0 ], targets: [0.0d] ],
                             [ inputs: [ 0, 1 ], targets: [1.0d] ],
                             [ inputs: [ 1, 0 ], targets: [2.0d] ],
                             [ inputs: [ 1, 1 ], targets: [3.0d] ] ]
            100.times {
                int sampleNo = 0
                dataSet.forEach( d -> {
                    net.train( d.inputs, d.targets, sampleNo++ )
                })
            }

        then:
            dataSet.forEach( d -> {
                List<Double> predictedValues = net.predict(d.inputs)
                System.out.println("expected: " + d.targets[0] + ". predicted: " + predictedValues[0])
                d.targets[0] == predictedValues[0]
            })
    }

    def 'train test 2: 4 layer network can be trained to change binary input to 1 if number is odd and to 0 if even'()
    {
        given:
            final NeuralNetwork net = new NeuralNetwork(0.5d, WeightInitializationType.NONE)
            net.addLayer(2, "I", null, null)
            net.addLayer(4, "H1", 0.3d, Activation.TANH)
            net.addLayer(3, "H2", 0.2d, Activation.TANH)
            net.addLayer(1, "O", 0.1d, Activation.TANH)
            net.initialize()
            net.visualize()

        when:
            List dataSet = [ [ inputs: [ 0, 0 ], targets: [0.0d] ],
                             [ inputs: [ 0, 1 ], targets: [1.0d] ],
                             [ inputs: [ 1, 0 ], targets: [0.0d] ],
                             [ inputs: [ 1, 1 ], targets: [1.0d] ] ]
            200.times {
                int sampleNo = 0
                dataSet.forEach( d -> {
                    net.train( d.inputs, d.targets, sampleNo++ )
                })
            }

        then:
            dataSet.forEach( d -> {
                List<Double> predictedValues = net.predict(d.inputs)
                System.out.println("expected: " + d.targets[0] + ". predicted: " + predictedValues[0])
                d.targets[0] == predictedValues[0]
            })
    }


    def 'serialization and deserialization works as expected'()
    {
        given:
            final NeuralNetwork ann = new NeuralNetwork(0.2d, WeightInitializationType.NONE)
            ann.addLayer(5, "I", null, null)
            ann.addLayer(3, "H", 0.2d, Activation.SIGMOID)
            ann.addLayer(1, "O", null, Activation.TANH)
        and:
            final String annFileName = ann.serializeToFile()

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
//                            LossFunctions.LossFunction.SQUARED_LOSS)
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
