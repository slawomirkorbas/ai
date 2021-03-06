package com.ai.tictactoe.model.neuralnetwork.general;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Example implements Serializable
{
    List<Double> inputs;
    List<Double> targets;

    public String toString()
    {
        String s = "";
        for(Double i : inputs) {
             s += String.valueOf(i);
        }
        return s;
    }
}
