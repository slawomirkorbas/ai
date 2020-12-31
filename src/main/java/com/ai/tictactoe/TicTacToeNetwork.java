package com.ai.tictactoe;

import com.ai.tictactoe.model.neuralnetwork.general.ActivationFunction;
import com.ai.tictactoe.model.neuralnetwork.general.NeuralNetwork;
import java.util.List;

public class TicTacToeNetwork
{
    final NeuralNetwork ann;

    public TicTacToeNetwork()
    {
        // Build test ANN network
        ann = new NeuralNetwork(0.2d);
        ann.addLayer(18, "I", null, null);
        ann.addLayer(15, "H", 0.1d, ActivationFunction.SIGMOID);
        ann.addLayer(9, "O", 0.1d, ActivationFunction.SIGMOID);
    }

    public void train(List games)
    {
        //for(Game g : games)
        //{
        //    for(int m=0; i<g.moves.size(); m++)
        //    {
        //        List<Double> targets = new ArrayList<>(9);
        //        for(int t=0; t<targets.size(); t++)
        //        {
        //            int targetFieldIndex = moveCoordinatesToFieldIndex(g.moves[m].row, g.moves[m].col);
        //            targets.get(i) = i == targetFieldIndex ? 1.0d : 0.0;
        //        }
        //        List<Integer> inputs = new ArrayList<>(18);
        //
        //        ann.train(inputs, targets);
        //    }
        //}

    }
}
