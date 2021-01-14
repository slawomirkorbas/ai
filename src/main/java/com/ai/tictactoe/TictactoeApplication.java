package com.ai.tictactoe;

import com.ai.tictactoe.game.BoardCell;
import com.ai.tictactoe.game.GameResult;
import com.ai.tictactoe.game.TicTacToeAgent;
import com.ai.tictactoe.game.TicTacToeGame;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SpringBootApplication
public class TictactoeApplication
{

    public static void main(String[] args)
    {
        SpringApplication.run(TictactoeApplication.class, args);
    }


    /**
     * Confronts two TicTacToe agents so they play multiple games. Each game is stored along with
     * all states(boards) and final result.
     * @param playerX - first agent playing with "x"
     * @param playerO - second agent playing with "o"
     * @param maxBatchSize - maximum games to play
     * @return
     */
    public static List<TicTacToeGame> generateGames(final TicTacToeAgent playerX,
                                             final TicTacToeAgent playerO,
                                             final Integer maxBatchSize)
    {
        final Map<String, TicTacToeGame> uniqueGamesMap = new HashMap<>();
        Integer total = 0;
        while(++total <= maxBatchSize)
        {
            GameResult result = GameResult.CONTINUE;
            TicTacToeGame newGame = new TicTacToeGame();
            TicTacToeAgent currentPlayer = playerO;
            while(!newGame.isFinished())
            {
                currentPlayer = currentPlayer == playerX ? playerO : playerX; // switch player
                BoardCell nextMove = currentPlayer.getNextMove(newGame.board);
                newGame.board[nextMove.row][nextMove.col] = currentPlayer.playAs;
                result = currentPlayer.gameState(newGame.board);
                newGame.update(result, nextMove);
            }
            if(result == GameResult.WIN)
            {
                newGame.whoWon = currentPlayer.playAs;
            }
            uniqueGamesMap.put(newGame.generateKey(), newGame);
            System.out.println("Played: " + total + ", unique stored: " + uniqueGamesMap.size());
        }
        return new ArrayList<>(uniqueGamesMap.values());
    }
}
