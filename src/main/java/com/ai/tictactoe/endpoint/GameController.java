package com.ai.tictactoe.endpoint;

import com.ai.tictactoe.TicTacToeNetwork;
import com.ai.tictactoe.dto.BoardDto;
import com.ai.tictactoe.util.BoardCell;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
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
    TicTacToeNetwork ticTacToeNetwork;

    @GetMapping("/tictactoe/newBoard")//, produces = MediaType.APPLICATION_JSON_VALUE)
    BoardDto getNewGame()
    {
        return new BoardDto();
    }

    @PostMapping("/tictactoe/predict")//(name = "/tictactoe/predict", produces = MediaType.APPLICATION_JSON_VALUE)
    BoardDto predictNextMove(@RequestBody BoardDto boardDto, @RequestParam String userFigure)
    {
        BoardCell nextMoveCell = ticTacToeNetwork.predictNextMove(boardDto.board);
        boardDto.board[nextMoveCell.getRow()][nextMoveCell.getCol()] = userFigure.equals("x") ? "o" : "x";
        return boardDto;
    }
}
