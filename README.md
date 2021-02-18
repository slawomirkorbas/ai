# Description
Spring Boot web application implementing well known Tic Tac Toe game. The idea of this application was to test
custom neural network (deep learning) library. 

# Configuration
The `TicTacToeConfiguration.java` contains `annTicTacToeAgent` bean definition which loads `net-*.ann` file
representing pre-trained neural network eg. `net-18-15-12-9-20210120-0840-batch-size-630-epochs-1418.ann`. 
Then each next move is predicted using this network by executing `forward-pass` algorithm for given board state.

# Training the neural network
`net-*ann` files for specific artificial neural network implementation can be created by `TicTacToeMachineLearningSpec.groovy` using
`File based supervised learning using JSON file with stored tic-tac-toe plays` test case. Here you can define your custom
neural network topology and decide which game batch files (JSON) will be used for the training session.

# Generating game JSON files for supervised training
Training files for Tic-Tac-Toe game can be generated using `GameGeneratorSpec` class by executing test case
`generateGames: play tic-tac-toe games using various agents and store results in JSON file`


# Example of Artificial Neural Network definition

```groovy
NeuralNetwork ann = nnf.build()
        .input(18, "I")
        .hidden(15, "H1", 0.1d, TransferFunction.TANH)
        .hidden(12, "H2", 0.1d, TransferFunction.TANH)
        .output(9 , "O" , 0.1d, TransferFunction.SOFTMAX, CostFunction.CROSS_ENTROPY)
        .initialize(WeightInitType.RANDOM)
```


