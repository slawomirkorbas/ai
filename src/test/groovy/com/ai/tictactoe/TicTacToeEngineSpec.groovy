package com.ai.tictactoe

import com.ai.tictactoe.game.MinMaxTicTacToeAgent
import com.ai.tictactoe.game.RandomTicTacToeAgent
import com.ai.tictactoe.game.TicTacToeAgent
import com.ai.tictactoe.game.TicTacToeGame
import com.fasterxml.jackson.databind.ObjectMapper
import spock.lang.Specification
import spock.lang.Unroll

class TicTacToeEngineSpec extends Specification {

    @Unroll
    def "generateGames: plays MinMax vs Random and store games in JSON format"() throws IOException
    {
        given:
            TicTacToeEngine engine = new TicTacToeEngine()
            ObjectMapper mapper = new ObjectMapper()

        when:
            List<TicTacToeGame> totalGames = engine.generateGames(playerX, playerO, 10000)
        then:
            totalGames.size() > 0
        and:
            // save to file as JSON
            File file = new File("game-batch-" + gameName + "-" + totalGames.size() + ".json")
            mapper.writeValue(file, totalGames)

        where:
            playerX                          | playerO                       | gameName
            new MinMaxTicTacToeAgent("x")    | new MinMaxTicTacToeAgent("o") | "mm-vs-mm"
            new MinMaxTicTacToeAgent("x")    | new RandomTicTacToeAgent("o") | "mm-vs-rnd"
            new RandomTicTacToeAgent("x")    | new RandomTicTacToeAgent("o") | "rnd-vs-rnd"
    }
}
