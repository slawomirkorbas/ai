package com.ai.tictactoe

import com.ai.tictactoe.game.GameGenerator
import com.ai.tictactoe.game.MinMaxTicTacToeAgent
import com.ai.tictactoe.game.RandomTicTacToeAgent
import com.ai.tictactoe.game.TicTacToeGame
import com.fasterxml.jackson.databind.ObjectMapper
import spock.lang.Specification
import spock.lang.Unroll

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter


class GameGeneratorSpec extends Specification
{

    @Unroll
    def "generateGames: play tic-tac-toe games using various agents and store results in JSON file"() throws IOException
    {
        given:
            ObjectMapper mapper = new ObjectMapper()
            GameGenerator generator = new GameGenerator()

        when:
            List<TicTacToeGame> totalGames = generator.generateGames(playerX, playerO, 50000)
        then:
            totalGames.size() > 0
        and:
            // save to file as JSON
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd-HHmm")
            File file = new File("game-batches-json/" + formatter.format(LocalDateTime.now()) + "-game-batch-" + gameName + "-" + totalGames.size() + ".json")
            mapper.writeValue(file, totalGames)

        where:
            playerX                          | playerO                       | gameName
            //new MinMaxTicTacToeAgent("x") | new MinMaxTicTacToeAgent("o") | "mmX-vs-mmO"
            //new MinMaxTicTacToeAgent("x")    | new RandomTicTacToeAgent("o") | "mmX-vs-rndO"
            //new MinMaxTicTacToeAgent("o")    | new RandomTicTacToeAgent("x") | "mmO-vs-rndX"
            new RandomTicTacToeAgent("x")    | new RandomTicTacToeAgent("o") | "rndX-vs-rndO"
            new RandomTicTacToeAgent("o")    | new RandomTicTacToeAgent("x") | "rndO-vs-rndX"
    }
}
