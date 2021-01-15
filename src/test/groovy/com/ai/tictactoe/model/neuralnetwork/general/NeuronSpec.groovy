package com.ai.tictactoe.model.neuralnetwork.general

import org.jgrapht.graph.DefaultWeightedEdge
import org.jgrapht.graph.SimpleDirectedWeightedGraph
import spock.lang.Specification
import spock.lang.Unroll

class NeuronSpec extends Specification
{
    @Unroll
    def "calcNetValue: works as expected"()
    {
        given:
            Neuron n = new Neuron("O1", TransferFunction.SIGMOID)
            SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> graph = new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class)
            graph.addVertex(n)
        and:
            Neuron p1 = new Neuron("H1", TransferFunction.SIGMOID)
            Neuron p2 = new Neuron("H2", TransferFunction.SIGMOID)
            graph.addVertex(p1)
            graph.addVertex(p2)
            DefaultWeightedEdge e1 = graph.addEdge(p1,n)
            DefaultWeightedEdge e2 = graph.addEdge(p2,n)
        and:
            graph.setEdgeWeight(e1, w1)
            graph.setEdgeWeight(e2, w2)
            p1.setOutputValue(o1)
            p2.setOutputValue(o2)

        expect:
            netTotal == n.calcNetValue(graph)

        where:
            o1   | o2   | w1  |  w2  |  netTotal
            1.0  | 2.0  | 0.5 |  1.0 |  2.5
            0.5  | 0.2  | 0.1 |  0.1 |  0.07
            -5.0 | 1.0  | 0.2 |  0.4 | -0.6
            0.0  | 0.0  | 0.5 | -1.0 |  0.0
    }


    @Unroll
    def "activate with SIGMOID: works as expected"()
    {
        given:
            Neuron n = new Neuron("O1", TransferFunction.SIGMOID)
            SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> graph = new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class)
            graph.addVertex(n)
        and:
            Neuron p1 = new Neuron("H1", TransferFunction.RELU)
            Neuron p2 = new Neuron("H2", TransferFunction.RELU)
            graph.addVertex(p1)
            graph.addVertex(p2)
            DefaultWeightedEdge e1 = graph.addEdge(p1,n)
            DefaultWeightedEdge e2 = graph.addEdge(p2,n)
        and:
            graph.setEdgeWeight(e1, w1)
            graph.setEdgeWeight(e2, w2)
            p1.setOutputValue(o1)
            p2.setOutputValue(o2)

        when:
            Double netTotal = n.calcNetValue(graph)
        then:
            (1/(1 + Math.exp(-netTotal))) == n.activate()

        where:
            o1   | o2   | w1  |  w2
            1.0  | 2.0  | 0.5 |  1.0
            0.5  | 0.2  | 0.1 |  0.1
            -5.0 | 1.0  | 0.2 |  0.4
            0.0  | 0.0  | 0.5 | -1.0
    }

    @Unroll
    def "activate with TANH: works as expected"()
    {
        given:
            Neuron n = new Neuron("O1", TransferFunction.TANH)
            SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> graph = new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class)
            graph.addVertex(n)
        and:
            Neuron p1 = new Neuron("H1", TransferFunction.RELU)
            Neuron p2 = new Neuron("H2", TransferFunction.RELU)
            graph.addVertex(p1)
            graph.addVertex(p2)
            DefaultWeightedEdge e1 = graph.addEdge(p1,n)
            DefaultWeightedEdge e2 = graph.addEdge(p2,n)
        and:
            graph.setEdgeWeight(e1, w1)
            graph.setEdgeWeight(e2, w2)
            p1.setOutputValue(o1)
            p2.setOutputValue(o2)

        when:
            Double netTotal = n.calcNetValue(graph)
        then:
            (Math.tanh(netTotal)) == n.activate()

        where:
            o1   | o2   | w1  |  w2
            1.0  | 2.0  | 0.5 |  1.0
            0.5  | 0.2  | 0.1 |  0.1
            -5.0 | 1.0  | 0.2 |  0.4
            0.0  | 0.0  | 0.5 | -1.0
    }

    @Unroll
    def "activate with RELU: works as expected"()
    {
        given:
            Neuron n = new Neuron("O1", TransferFunction.RELU)
            SimpleDirectedWeightedGraph<Neuron, DefaultWeightedEdge> graph = new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class)
            graph.addVertex(n)
        and:
            Neuron p1 = new Neuron("H1", TransferFunction.RELU)
            Neuron p2 = new Neuron("H2", TransferFunction.RELU)
            graph.addVertex(p1)
            graph.addVertex(p2)
            DefaultWeightedEdge e1 = graph.addEdge(p1,n)
            DefaultWeightedEdge e2 = graph.addEdge(p2,n)
        and:
            graph.setEdgeWeight(e1, w1)
            graph.setEdgeWeight(e2, w2)
            p1.setOutputValue(o1)
            p2.setOutputValue(o2)

        when:
            Double netTotal = n.calcNetValue(graph)
        then:
            Math.max(0.0, netTotal) == n.activate()

        where:
            o1   | o2   | w1  |  w2
            1.0  | 2.0  | 0.5 |  1.0
            0.5  | 0.2  | 0.1 |  0.1
            -5.0 | 1.0  | 0.2 |  0.4
            0.0  | 0.0  | 0.5 | -1.0
    }

    def "activate with SOFTMAX: works as expected"()
    {
        given:
            Neuron n = new Neuron("O1", TransferFunction.SOFTMAX)
            n.setNetVal(netVal)
        and:
            // potential  "netto" values from all the neurons in the same layer
            List<Double> netVals = [ netVal, 0.4d, 5.0d, 1.25d ]
        and:
            Double maxNetVal = 5.0d //Collections.max(netVals)
            Double totalExpSum = netVals.stream()
                .map(v -> Math.exp(v - maxNetVal))
                .reduce(0.0, Double::sum)

        expect:
            (Math.exp(n.getNetVal() - maxNetVal)/totalExpSum).equals(n.activateWithSoftmax(netVals))

        where:
            netVal << [ 1.0d, 0.0d, -0.25d, -5.0d, 4.0d ]

    }
}
