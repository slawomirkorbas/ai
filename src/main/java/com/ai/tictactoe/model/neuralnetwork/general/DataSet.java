package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class DataSet implements Serializable
{
    List<Example> examples = new ArrayList<>();

    public void addExample(List<Double> inputs, List<Double> targets)
    {
        examples.add(new Example(inputs, targets));
    }

    public int size()
    {
        return examples.size();
    }

    public Example example(int index)
    {
        return examples.get(index);
    }
}
