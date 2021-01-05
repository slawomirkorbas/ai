package com.ai.tictactoe.model.neuralnetwork.general

import spock.lang.Specification

class DerivativesSpec extends Specification
{
    def "derivatives are calculated correctly"()
    {
        given:
            Double tanhValue = ActivationFunction.TANH.apply(1.0d)
            assert(tanhValue == 0.7615941559557647d)

        expect:
            0.4199743416140257d == Derivatives.getDerivative(ActivationFunction.TANH).apply(tanhValue)
    }
}
