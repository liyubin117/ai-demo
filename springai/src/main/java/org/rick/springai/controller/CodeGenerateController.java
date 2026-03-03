package org.rick.springai.controller;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.deepseek.DeepSeekAssistantMessage;
import org.springframework.ai.deepseek.DeepSeekChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class CodeGenerateController {

    private final DeepSeekChatModel chatModel;

    @Autowired
    public CodeGenerateController(DeepSeekChatModel chatModel) {
        this.chatModel = chatModel;
    }

    // 使用后发现没生效。。但用的是官网的例子
    @GetMapping("/generatePythonCode")
    public String generate(@RequestParam(value = "message", defaultValue = "Please write quick sort code") String message) {
        UserMessage userMessage = new UserMessage(message);
        Message assistantMessage = DeepSeekAssistantMessage.prefixAssistantMessage("```python\\n");
        Prompt prompt = new Prompt(
                List.of(userMessage, assistantMessage)
                ,ChatOptions.builder().stopSequences(List.of("```")).build());
        ChatResponse response = chatModel.call(prompt);
        return response.getResult().getOutput().getText();
    }
}