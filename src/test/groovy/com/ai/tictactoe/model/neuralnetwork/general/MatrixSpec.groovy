package com.ai.tictactoe.model.neuralnetwork.general


import spock.lang.Specification

class MatrixSpec extends Specification
{

    def 'applyFunction: sigmoid function returns expected result'()
    {
        given:
        Matrix m = new Matrix([[0.0d, 0.5d, 0.6d ],
                               [0.7d, 0.8d, 0.9d ],
                               [2.0d, 2.5d, 3.0d ] ])
        and:
            Matrix expected = new Matrix([ [0.50000d, 0.62246d, 0.64566d ],
                                           [0.66819d, 0.68997d, 0.71095d ],
                                           [0.88080d, 0.92414d, 0.95257d ] ])

        when:
            m.applyFunction(ActivationFunction.SIGMOID)

        then:
            m.equals(expected)
    }


    def 'applyFunction: tanh function returns expected result'()
    {
        given:
            Matrix m = new Matrix([ [0.0d, 0.4d, 0.8d ],
                                    [1.2d, 1.6d, 2.0d ] ])
        and:
            Matrix expected = new Matrix([ [0.00000d, 0.37994d, 0.66403d ],
                                           [0.83365d, 0.92167d, 0.96402d ] ])

        when:
            m.applyFunction(ActivationFunction.TANH)

        then:
            m.equals(expected)
    }


    def 'applyFunction: derivative tanh function returns expected result'()
    {
        given:
            Matrix m = new Matrix([ [0.1d ],
                                    [2.0d ] ])
        and:
            Matrix expected = new Matrix([ [ 1 - Math.pow(Math.tanh(0.1d), 2) ],
                                           [ 1 - Math.pow(Math.tanh(2.0d), 2) ] ])

        when:
            m.applyFunction(ActivationFunction.TANH)
            m.applyFunction(Derivatives.DTANH)

        then:
            m.equals(expected)
    }
}
