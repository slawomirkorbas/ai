package com.ai.tictactoe.endpoint;

import com.ai.tictactoe.game.AnnTicTacToeAgent;
import com.ai.tictactoe.dto.BoardDto;
import com.ai.tictactoe.game.BoardCell;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Default Game controller class.
 *
 * @author Slawomir Korbas
 */
@RestController
public class GameController
{

    @Autowired
    AnnTicTacToeAgent annTicTacToeAgent;

    @GetMapping("/tictactoe/newBoard")//, produces = MediaType.APPLICATION_JSON_VALUE)
    BoardDto getNewGame()
    {
        return new BoardDto();
    }

    @PostMapping("/tictactoe/predict")//(name = "/tictactoe/predict", produces = MediaType.APPLICATION_JSON_VALUE)
    BoardDto predictNextMove(@RequestBody BoardDto boardDto, @RequestParam String userFigure)
    {
        BoardCell nextMoveCell = annTicTacToeAgent.getNextMove(boardDto.board);
        boardDto.board[nextMoveCell.row][nextMoveCell.col] = userFigure.equals("x") ? "o" : "x";
        return boardDto;
    }
}
