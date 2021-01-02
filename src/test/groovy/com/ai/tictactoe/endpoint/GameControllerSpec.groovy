package com.ai.tictactoe.endpoint

import com.fasterxml.jackson.databind.ObjectMapper
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.annotation.Rollback
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.ResultActions
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import org.springframework.web.context.WebApplicationContext
import spock.lang.Specification
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status



@SpringBootTest
@Rollback
class GameControllerSpec extends Specification
{
    MockMvc mockMvc

    @Autowired
    protected WebApplicationContext webApplicationContext

    @Autowired
    ObjectMapper objectMapper

    def setup()
    {
        mockMvc = MockMvcBuilders.webAppContextSetup(webApplicationContext).build()
    }

    def "newBoard: returns an empty board"()
    {
        when:
            ResultActions result = mockMvc.perform(MockMvcRequestBuilders.get("/tictactoe/newBoard"))

        then:
            result.andExpect(status().isOk())
            //.andExpect(content().json("{matrix:...}}"))
    }
}
