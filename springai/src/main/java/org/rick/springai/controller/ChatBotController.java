package org.rick.springai.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import java.text.SimpleDateFormat;
import java.util.Date;

@RestController
@RequestMapping("/weather")
public class ChatBotController {

    private final ChatClient chatClient;

    public ChatBotController(ChatClient.Builder builder) {
        this.chatClient = builder.defaultSystem("你是一个天气预报员，当有人输入日期的时候，你输出上海的天气预报信息，" +
                "生成结果在html页面中以markdown的格式输出，还要注意换行格式友好，最后输出结尾的时候始终以下面的语句结尾：感谢您的咨询，我是舆情君rick。").build();
    }

    @GetMapping({"/{message}", "/"})
    public String chat(@PathVariable(value = "message", required = false) String message) {
        try {
            if (!StringUtils.hasText(message)) {
                message = new SimpleDateFormat().format(new Date());
            }
            return chatClient.prompt()
                    .user(message)
                    .call()
                    .content();
        } catch (Exception e) {
            // 返回友好的错误信息
            return ("抱歉，聊天服务暂时不可用: " + e.getMessage());
        }
    }
}