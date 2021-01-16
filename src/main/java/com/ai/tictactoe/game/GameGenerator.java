package com.ai.tictactoe.game;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GameGenerator
{
    /**
     * Confronts two TicTacToe agents so they play multiple games. Each game is stored along with
     * all states(boards) and final result.
     * @param playerX - first agent playing with "x"
     * @param playerO - second agent playing with "o"
     * @param maxBatchSize - maximum games to play
     * @return
     */
    public List<TicTacToeGame> generateGames(final TicTacToeAgent playerX,
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
                BoardCell cell = currentPlayer.getNextMove(newGame.board);
                newGame.board[cell.row][cell.col] = currentPlayer.playAs;
                result = currentPlayer.gameState(newGame.board);
                newGame.update(result, new Move(cell, currentPlayer.playAs));
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
