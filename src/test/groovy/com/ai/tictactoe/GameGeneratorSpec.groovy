package com.ai.tictactoe

import com.ai.tictactoe.game.GameGenerator
import com.ai.tictactoe.game.RandomTicTacToeAgent
import com.ai.tictactoe.game.TicTacToeGame
import com.fasterxml.jackson.databind.ObjectMapper
import spock.lang.Specification
import spock.lang.Unroll


class GameGeneratorSpec extends Specification
{

    @Unroll
    def "generateGames: plays MinMax vs Random and store games in JSON format"() throws IOException
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
            File file = new File("game-batch-" + gameName + "-" + totalGames.size() + ".json")
            mapper.writeValue(file, totalGames)

        where:
            playerX                          | playerO                       | gameName
            //new MinMaxTicTacToeAgent("x")    | new MinMaxTicTacToeAgent("o") | "mmX-vs-mmO"
            //new MinMaxTicTacToeAgent("x")    | new RandomTicTacToeAgent("o") | "mmX-vs-rndO"
            new RandomTicTacToeAgent("x")    | new RandomTicTacToeAgent("o") | "rndX-vs-rndO"
    }
}
