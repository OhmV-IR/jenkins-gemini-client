package io.ohmvir.plugins.geminiclient;

import static org.junit.jupiter.api.Assertions.*;

import com.google.genai.types.Blob;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import com.google.genai.types.Type;
import hudson.model.Descriptor;
import io.ohmvir.plugins.jenkinsaisynapse.api.ModelConversation;
import io.ohmvir.plugins.jenkinsaisynapse.api.input.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.ModelFinishReason;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.ModelThinkingLevel;
import io.ohmvir.plugins.jenkinsaisynapse.api.output.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.skills.SkillData;
import io.ohmvir.plugins.jenkinsaisynapse.api.tools.Tool;
import io.ohmvir.plugins.jenkinsaisynapse.api.tools.ToolArgumentDescription;
import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class GeminiClientStepTest {
    private static final String TEXT_RESPONSE = """
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "%s"}]}, "finishReason": "STOP"}]}
            """;

    /** Replaces the network round trip with queued responses and records every request. */
    static class FakeGeminiClient extends GeminiClient {
        final Deque<Object> responses = new ArrayDeque<>();
        final List<List<Content>> sentContents = new ArrayList<>();
        final List<GenerateContentConfig> sentConfigs = new ArrayList<>();

        FakeGeminiClient respond(String json) {
            responses.add(GenerateContentResponse.fromJson(json));
            return this;
        }

        FakeGeminiClient respondText(String text) {
            return respond(TEXT_RESPONSE.formatted(text));
        }

        FakeGeminiClient fail(RuntimeException e) {
            responses.add(e);
            return this;
        }

        @Override
        GenerateContentResponse generateContent(
                GeminiModelSettings configuration,
                GeminiClientSettings clientConfiguration,
                List<Content> contents,
                GenerateContentConfig config) {
            sentContents.add(contents);
            sentConfigs.add(config);
            Object next = responses.poll();
            if (next == null) throw new AssertionError("Unexpected Gemini API call");
            if (next instanceof RuntimeException e) throw e;
            return (GenerateContentResponse) next;
        }

        List<Content> lastContents() {
            return sentContents.getLast();
        }

        GenerateContentConfig lastConfig() {
            return sentConfigs.getLast();
        }
    }

    public static class WeatherTool extends Tool {
        public WeatherTool() throws NoSuchMethodException {}

        @Override
        public String getName() {
            return "weather";
        }

        @Override
        public String getDescription() {
            return "Gets the weather forecast";
        }

        @Override
        public List<ToolArgumentDescription> getArguments() {
            return List.of(
                    new ToolArgumentDescription("city", String.class, "City name", true),
                    new ToolArgumentDescription("days", int.class, "Forecast length", false));
        }

        public String weather(String city, int days) {
            return city + ":" + days;
        }
    }

    public static class ClockTool extends Tool {
        public ClockTool() throws NoSuchMethodException {}

        @Override
        public String getName() {
            return "clock";
        }

        @Override
        public String getDescription() {
            return "Gets the current time";
        }

        @Override
        public List<ToolArgumentDescription> getArguments() {
            return List.of();
        }

        public String clock() {
            return "noon";
        }
    }

    private FakeGeminiClient client;
    private GeminiModelSettings model;
    private GeminiClientSettings clientSettings;
    private ModelRequest request;

    @BeforeEach
    void setUp() throws Descriptor.FormException {
        client = new FakeGeminiClient();
        model = new GeminiModelSettings("gemini-2.5-flash", "gemini-key");
        clientSettings = new GeminiClientSettings(60L, 1000L);
        request = new ModelRequest();
    }

    private List<ModelOutput> step() {
        return client.takeStep(null, model, clientSettings, request, request.getModelInputs());
    }

    /** Mirrors ModelRequest.execute: answer every tool call and keep stepping until a step returns nothing. */
    private List<ModelOutput> execute(Tool tool) {
        List<ModelOutput> outputs = new ArrayList<>();
        for (int steps = 0; steps < 10; steps++) {
            List<ModelOutput> stepOutputs = step();
            if (stepOutputs.isEmpty()) return outputs;
            outputs.addAll(stepOutputs);
            for (ModelOutput output : stepOutputs) {
                if (output instanceof ToolCallContent call) {
                    request.addInput(new ToolCallResponseContent(
                            call.getToolUseId(), true, "result of " + call.getName(), tool));
                }
            }
        }
        throw new AssertionError("Request did not finish");
    }

    private static <T> List<T> outputsOf(List<ModelOutput> outputs, Class<T> type) {
        return outputs.stream().filter(type::isInstance).map(type::cast).toList();
    }

    private static List<ModelFinishReason> finishReasons(List<ModelOutput> outputs) {
        return outputsOf(outputs, FinishReasonContent.class).stream()
                .map(FinishReasonContent::getReason)
                .toList();
    }

    private static String role(Content content) {
        return content.role().orElseThrow();
    }

    private static List<Part> parts(Content content) {
        return content.parts().orElseThrow();
    }

    private static FunctionResponse functionResponse(Part part) {
        return part.functionResponse().orElseThrow();
    }

    private static List<String> roles(List<Content> contents) {
        return contents.stream().map(GeminiClientStepTest::role).toList();
    }

    @Test
    public void textResponseProducesTextUsageAndStop() {
        request.addInput(new SystemPromptContent("Be brief."));
        request.addInput(new InputSkillContent(
                new SkillData("skill", "A skill", null, null, null, null, "Skill text", Map.of())));
        request.addInput(new InputTextContent("Hello"));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hi there"}]}, "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 3,
                                   "cachedContentTokenCount": 2}}
                """);

        List<ModelOutput> outputs = execute(null);

        assertEquals(1, client.sentContents.size(), "Synapse's follow-up step should not call the API again");
        assertEquals(
                List.of("Hi there"),
                outputsOf(outputs, OutputTextContent.class).stream()
                        .map(OutputTextContent::getText)
                        .toList());
        assertEquals(List.of(ModelFinishReason.STOP), finishReasons(outputs));
        TokenUtilizationContent usage =
                outputsOf(outputs, TokenUtilizationContent.class).getFirst();
        assertEquals(10, usage.getInputTokensUsed());
        assertEquals(8, usage.getOutputTokensUsed(), "Thinking tokens are billed as output");
        assertEquals(2, usage.getCachedTokensUsed());

        List<Content> contents = client.lastContents();
        assertEquals(List.of("user"), roles(contents));
        assertEquals("Hello", parts(contents.getFirst()).getFirst().text().orElseThrow());
        GenerateContentConfig config = client.lastConfig();
        assertEquals(
                "Be brief.\n\nSkill text",
                parts(config.systemInstruction().orElseThrow())
                        .getFirst()
                        .text()
                        .orElseThrow());
        assertEquals(1000, config.maxOutputTokens().orElseThrow());
        assertTrue(config.tools().isEmpty());
        assertTrue(config.thinkingConfig().isEmpty());
    }

    @Test
    public void thoughtPartsBecomeThinkingContentAndEmptyTextIsSkipped() {
        request.addInput(new InputTextContent("Think"));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"text": "Pondering", "thought": true},
                    {"text": "", "thoughtSignature": "c2ln"},
                    {"text": "Answer"}]}, "finishReason": "STOP"}]}
                """);

        List<ModelOutput> outputs = step();

        assertEquals(
                List.of("Pondering"),
                outputsOf(outputs, ThinkingContent.class).stream()
                        .map(ThinkingContent::getThinking)
                        .toList());
        assertEquals(
                List.of("Answer"),
                outputsOf(outputs, OutputTextContent.class).stream()
                        .map(OutputTextContent::getText)
                        .toList());
    }

    @Test
    public void safetyFinishReasonIsReportedInsteadOfThrown() {
        request.addInput(new InputTextContent("Something unsafe"));
        client.respond("""
                {"candidates": [{"finishReason": "SAFETY"}]}
                """);

        assertEquals(List.of(ModelFinishReason.SAFEGUARD), finishReasons(step()));
        assertTrue(step().isEmpty());
    }

    @Test
    public void recitationKeepsPartialTextAndReportsSafeguard() {
        request.addInput(new InputTextContent("Recite"));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [{"text": "Partial"}]}, "finishReason": "RECITATION"}]}
                """);

        List<ModelOutput> outputs = step();

        assertEquals(1, outputsOf(outputs, OutputTextContent.class).size());
        assertEquals(List.of(ModelFinishReason.SAFEGUARD), finishReasons(outputs));
    }

    @Test
    public void maxTokensIsReportedAsTokenCapEvenWithoutParts() {
        request.addInput(new InputTextContent("Long answer please"));
        client.respond("""
                {"candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}]}
                """);

        assertEquals(List.of(ModelFinishReason.TOKEN_CAP), finishReasons(step()));
    }

    @Test
    public void blockedPromptIsReportedAsSafeguard() {
        request.addInput(new InputTextContent("Blocked"));
        client.respond("""
                {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": {"promptTokenCount": 4}}
                """);

        List<ModelOutput> outputs = step();

        assertEquals(List.of(ModelFinishReason.SAFEGUARD), finishReasons(outputs));
        assertEquals(
                4, outputsOf(outputs, TokenUtilizationContent.class).getFirst().getInputTokensUsed());
    }

    @Test
    public void finishReasonsMapToSynapseReasons() {
        assertEquals(
                ModelFinishReason.STOP, GeminiClient.finishReason(new com.google.genai.types.FinishReason("STOP")));
        assertEquals(
                ModelFinishReason.TOKEN_CAP,
                GeminiClient.finishReason(new com.google.genai.types.FinishReason("MAX_TOKENS")));
        for (String reason :
                List.of("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY")) {
            assertEquals(
                    ModelFinishReason.SAFEGUARD,
                    GeminiClient.finishReason(new com.google.genai.types.FinishReason(reason)),
                    reason);
        }
        assertEquals(
                ModelFinishReason.STOP,
                GeminiClient.finishReason(new com.google.genai.types.FinishReason("MALFORMED_FUNCTION_CALL")));
    }

    @Test
    public void responseWithoutCandidatesFailsAndCanBeRetried() {
        request.addInput(new InputTextContent("Hello"));
        client.respond("{}").respondText("Recovered");

        assertThrows(IllegalStateException.class, this::step);
        List<ModelOutput> retried = step();

        assertEquals(1, outputsOf(retried, OutputTextContent.class).size());
        assertEquals(2, client.sentContents.size());
    }

    @Test
    public void apiFailureIsWrappedAndCanBeRetried() {
        request.addInput(new InputTextContent("Hello"));
        client.fail(new RuntimeException("503 Service Unavailable")).respondText("Recovered");

        IllegalStateException failure = assertThrows(IllegalStateException.class, this::step);
        assertTrue(failure.getMessage().contains("503 Service Unavailable"));
        assertEquals(1, outputsOf(step(), OutputTextContent.class).size());
    }

    @Test
    public void toolCallRoundTripReplaysModelTurnAndSendsMatchingFunctionResponse() throws Exception {
        WeatherTool tool = new WeatherTool();
        request.addInput(new InputTextContent("Weather in Paris?"));
        request.addInput(new InputToolContent(tool));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"id": "call-1", "name": "weather", "args": {"city": "Paris", "days": 3}},
                     "thoughtSignature": "c2ln"}]}, "finishReason": "STOP"}]}
                """).respondText("Sunny");

        List<ModelOutput> outputs = execute(tool);

        ToolCallContent call = outputsOf(outputs, ToolCallContent.class).getFirst();
        assertEquals("call-1", call.getToolUseId());
        assertEquals("weather", call.getName());
        assertEquals("Paris", call.getToolArguments().get("city").getAsString());
        assertEquals(3, call.getToolArguments().get("days").getAsInt());
        assertEquals(List.of(ModelFinishReason.TOOL_CALLS, ModelFinishReason.STOP), finishReasons(outputs));

        List<Content> second = client.sentContents.get(1);
        assertEquals(List.of("user", "model", "user"), roles(second));
        Part replayedCall = parts(second.get(1)).getFirst();
        assertEquals("weather", replayedCall.functionCall().orElseThrow().name().orElseThrow());
        assertTrue(replayedCall.thoughtSignature().isPresent(), "Thought signatures must be sent back to Gemini");
        FunctionResponse response = functionResponse(parts(second.get(2)).getFirst());
        assertEquals("weather", response.name().orElseThrow());
        assertEquals("call-1", response.id().orElseThrow());
        assertEquals(Map.of("output", "result of weather"), response.response().orElseThrow());
    }

    @Test
    public void multipleToolRoundsReplayEveryModelTurnInOrder() throws Exception {
        WeatherTool tool = new WeatherTool();
        request.addInput(new InputTextContent("Compare Paris and Rome"));
        request.addInput(new InputToolContent(tool));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"id": "a", "name": "weather", "args": {"city": "Paris"}}}]}}]}
                """).respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"id": "b", "name": "weather", "args": {"city": "Rome"}}}]}}]}
                """).respondText("Both sunny");

        execute(tool);

        List<Content> third = client.sentContents.get(2);
        assertEquals(List.of("user", "model", "user", "model", "user"), roles(third));
        assertEquals(
                "a",
                parts(third.get(1)).getFirst().functionCall().orElseThrow().id().orElseThrow());
        assertEquals("a", functionResponse(parts(third.get(2)).getFirst()).id().orElseThrow());
        assertEquals(
                "b",
                parts(third.get(3)).getFirst().functionCall().orElseThrow().id().orElseThrow());
        assertEquals("b", functionResponse(parts(third.get(4)).getFirst()).id().orElseThrow());
    }

    @Test
    public void parallelFunctionResponsesAreGroupedIntoOneContent() throws Exception {
        WeatherTool tool = new WeatherTool();
        request.addInput(new InputTextContent("Paris and Rome?"));
        request.addInput(new InputToolContent(tool));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"id": "a", "name": "weather", "args": {"city": "Paris"}}},
                    {"functionCall": {"id": "b", "name": "weather", "args": {"city": "Rome"}}}]}}]}
                """).respondText("Done");

        execute(tool);

        List<Content> second = client.sentContents.get(1);
        assertEquals(List.of("user", "model", "user"), roles(second));
        List<Part> responses = parts(second.get(2));
        assertEquals(2, responses.size());
        assertEquals(
                List.of("a", "b"),
                responses.stream()
                        .map(part -> functionResponse(part).id().orElseThrow())
                        .toList());
    }

    @Test
    public void generatedToolUseIdsAreNotSentBackToGemini() throws Exception {
        WeatherTool tool = new WeatherTool();
        request.addInput(new InputTextContent("Weather?"));
        request.addInput(new InputToolContent(tool));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"name": "weather", "args": {"city": "Oslo"}}}]}}]}
                """).respondText("Cold");

        List<ModelOutput> outputs = execute(tool);

        assertFalse(outputsOf(outputs, ToolCallContent.class)
                .getFirst()
                .getToolUseId()
                .isBlank());
        FunctionResponse response =
                functionResponse(parts(client.lastContents().get(2)).getFirst());
        assertTrue(response.id().isEmpty());
        assertEquals("weather", response.name().orElseThrow());
    }

    @Test
    public void failedToolCallUsesCalledFunctionNameAndErrorKey() {
        request.addInput(new InputTextContent("Use a tool"));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"id": "x", "name": "missing_tool", "args": {}}}]}}]}
                """).respondText("Sorry");

        ToolCallContent call = outputsOf(step(), ToolCallContent.class).getFirst();
        // Synapse reports tools it cannot find with no response content and no tool.
        request.addInput(new ToolCallResponseContent(call.getToolUseId(), false, null, null));
        step();

        FunctionResponse response =
                functionResponse(parts(client.lastContents().get(2)).getFirst());
        assertEquals("missing_tool", response.name().orElseThrow());
        assertEquals(Map.of("error", "Tool call failed"), response.response().orElseThrow());
    }

    @Test
    public void toolsAreDeclaredInSingleGeminiToolWithValidSchemas() throws Exception {
        WeatherTool weather = new WeatherTool();
        request.addInput(new InputTextContent("Hi"));
        request.addInput(new InputToolContent(weather));
        request.addInput(new InputToolContent(new ClockTool()));
        request.addInput(new InputToolContent(weather));
        client.respondText("Hello");

        step();

        List<com.google.genai.types.Tool> tools = client.lastConfig().tools().orElseThrow();
        assertEquals(1, tools.size());
        List<FunctionDeclaration> declarations =
                tools.getFirst().functionDeclarations().orElseThrow();
        assertEquals(
                List.of("weather", "clock"),
                declarations.stream().map(d -> d.name().orElseThrow()).toList());

        FunctionDeclaration weatherDeclaration = declarations.get(0);
        assertEquals(
                "Gets the weather forecast", weatherDeclaration.description().orElseThrow());
        Schema parameters = weatherDeclaration.parameters().orElseThrow();
        assertEquals(Type.Known.OBJECT, parameters.type().orElseThrow().knownEnum());
        assertEquals(List.of("city"), parameters.required().orElseThrow());
        Map<String, Schema> properties = parameters.properties().orElseThrow();
        assertEquals(
                Type.Known.STRING, properties.get("city").type().orElseThrow().knownEnum());
        assertEquals("City name", properties.get("city").description().orElseThrow());
        assertEquals(
                Type.Known.INTEGER, properties.get("days").type().orElseThrow().knownEnum());

        assertTrue(declarations.get(1).parameters().isEmpty(), "Argument-less tools must not declare parameters");
    }

    @Test
    public void schemaTypesMatchGeminiRequirements() {
        assertEquals("INTEGER", GeminiClient.schemaType(long.class));
        assertEquals("INTEGER", GeminiClient.schemaType(Integer.class));
        assertEquals("NUMBER", GeminiClient.schemaType(double.class));
        assertEquals("NUMBER", GeminiClient.schemaType(Float.class));
        assertEquals("BOOLEAN", GeminiClient.schemaType(boolean.class));
        assertEquals("STRING", GeminiClient.schemaType(char.class));
        assertEquals("STRING", GeminiClient.schemaType(Object.class));

        Schema array = GeminiClient.schema(int[].class, "Numbers");
        assertEquals(Type.Known.ARRAY, array.type().orElseThrow().knownEnum());
        assertEquals(
                Type.Known.INTEGER,
                array.items().orElseThrow().type().orElseThrow().knownEnum());
        Schema list = GeminiClient.schema(List.class, null);
        assertEquals(
                Type.Known.STRING,
                list.items().orElseThrow().type().orElseThrow().knownEnum());
    }

    @Test
    public void conversationHistoryKeepsTurnOrderAndPrecedesCurrentTurn() throws Exception {
        ModelConversation conversation = new ModelConversation() {
            @Override
            public void save() {}
        };
        conversation.addAllContent(List.of(
                new InputTextContent("first question"),
                new SystemPromptContent("ignored"),
                new ThinkingContent("ignored"),
                new OutputTextContent("first answer"),
                new InputTextContent("second question"),
                new OutputTextContent("second answer")));
        request.addInput(new InputTextContent("new question"));
        request.addInput(new InputConversationContent(conversation));
        client.respondText("new answer");

        step();

        List<Content> contents = client.lastContents();
        assertEquals(List.of("user", "model", "user", "model", "user"), roles(contents));
        assertEquals(
                List.of("first question", "first answer", "second question", "second answer", "new question"),
                contents.stream()
                        .map(content -> parts(content).getFirst().text().orElseThrow())
                        .toList());
    }

    @Test
    public void imageInputIsSentAsPngBytes() {
        request.addInput(new InputImageContent(new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB)));
        client.respondText("A black square");

        step();

        Blob blob =
                parts(client.lastContents().getFirst()).getFirst().inlineData().orElseThrow();
        assertEquals("image/png", blob.mimeType().orElseThrow());
        byte[] data = blob.data().orElseThrow();
        assertArrayEquals(new byte[] {(byte) 0x89, 'P', 'N', 'G'}, Arrays.copyOf(data, 4));
    }

    @Test
    public void mediaInputsUseExpectedMimeTypes() {
        request.addInput(new InputFileContent("doc.pdf", new byte[] {1}, "application/pdf"));
        request.addInput(new InputVideoContent(new byte[] {2}));
        request.addInput(new InputAudioContent(new byte[] {3}));
        client.respondText("ok");

        step();

        assertEquals(
                List.of("application/pdf", "video/mp4", "audio/mpeg"),
                parts(client.lastContents().getFirst()).stream()
                        .map(part -> part.inlineData().orElseThrow().mimeType().orElseThrow())
                        .toList());
    }

    @Test
    public void generationParametersArePassedThrough() {
        request.addInput(new InputTextContent("Hi"));
        request.addInput(new TemperatureContent(0.5));
        request.addInput(new TopPContent(0.9));
        request.addInput(new StopSequencesContent(List.of("END")));
        request.addInput(new MaxOutputTokensContent(Long.MAX_VALUE));
        client.respondText("ok");

        step();

        GenerateContentConfig config = client.lastConfig();
        assertEquals(0.5f, config.temperature().orElseThrow());
        assertEquals(0.9f, config.topP().orElseThrow());
        assertEquals(List.of("END"), config.stopSequences().orElseThrow());
        assertEquals(Integer.MAX_VALUE, config.maxOutputTokens().orElseThrow());
    }

    @Test
    public void outOfRangeSamplingParametersAreRejected() {
        request.addInput(new TemperatureContent(2.5));
        assertThrows(IllegalArgumentException.class, this::step);

        request = new ModelRequest();
        request.addInput(new TopPContent(-0.1));
        assertThrows(IllegalArgumentException.class, this::step);
        assertTrue(client.sentContents.isEmpty());
    }

    @Test
    public void thinkingLevelIsSentAsThinkingConfig() {
        request.addInput(new InputTextContent("Hi"));
        request.addInput(new ThinkingLevelContent(ModelThinkingLevel.HIGH));
        client.respondText("ok");

        step();

        var thinking = client.lastConfig().thinkingConfig().orElseThrow();
        assertEquals(16384, thinking.thinkingBudget().orElseThrow());
        assertTrue(thinking.includeThoughts().orElseThrow());
        assertTrue(thinking.thinkingLevel().isEmpty(), "Gemini 2.5 rejects thinking levels");
    }

    @Test
    public void requestedOutputClassesFilterOutputsButKeepToolCalls() throws Exception {
        WeatherTool tool = new WeatherTool();
        request.requestOutputType(OutputTextContent.class);
        request.addInput(new InputTextContent("Weather?"));
        request.addInput(new InputToolContent(tool));
        client.respond("""
                {"candidates": [{"content": {"role": "model", "parts": [
                    {"text": "Checking", "thought": true},
                    {"functionCall": {"id": "a", "name": "weather", "args": {"city": "Paris"}}}]}}],
                 "usageMetadata": {"promptTokenCount": 1}}
                """);

        List<ModelOutput> outputs = step();

        assertEquals(1, outputs.size());
        assertInstanceOf(ToolCallContent.class, outputs.getFirst());
    }

    @Test
    public void earlierAnswersAreReplayedWhenRequestIsContinued() {
        request.addInput(new InputTextContent("Name a colour"));
        client.respondText("Blue").respondText("Red");
        execute(null);

        request.addInput(new InputTextContent("Another one"));
        execute(null);

        List<Content> contents = client.lastContents();
        assertEquals(List.of("user", "model", "user"), roles(contents));
        assertEquals("Blue", parts(contents.get(1)).getFirst().text().orElseThrow());
    }

    @Test
    public void requestStateIsTrackedPerRequest() {
        request.addInput(new InputTextContent("First"));
        client.respondText("One").respondText("Two");
        execute(null);

        request = new ModelRequest();
        request.addInput(new InputTextContent("Second"));
        execute(null);

        assertEquals(2, client.sentContents.size());
        assertEquals(List.of("user"), roles(client.lastContents()));
    }
}
