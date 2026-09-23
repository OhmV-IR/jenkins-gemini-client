package io.ohmvir.plugins.geminiclient;

import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.HttpOptions;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import com.google.genai.types.ThinkingConfig;
import com.google.genai.types.Tool;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import hudson.Extension;
import hudson.model.Descriptor;
import io.ohmvir.plugins.jenkinsaisynapse.api.client.ModelClient;
import io.ohmvir.plugins.jenkinsaisynapse.api.input.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.output.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.tools.ToolArgumentDescription;
import io.ohmvir.plugins.jenkinsaisynapse.utils.SecretsUtils;
import java.util.*;
import org.jspecify.annotations.NonNull;

@Extension
public class GeminiClient extends ModelClient<GeminiModelSettings, GeminiClientSettings> {
    private static final Gson GSON = new Gson();

    private static final class RequestState {
        int inputCount = -1;
        Content previousModelContent;
        boolean hadToolCalls;
    }

    private final Map<ModelRequest, RequestState> requestStates = Collections.synchronizedMap(new WeakHashMap<>());

    @Override
    protected List<ModelOutput> takeStepImpl(
            ModelData modelData,
            GeminiModelSettings configuration,
            GeminiClientSettings clientConfiguration,
            ModelRequest request,
            List<ModelInput> turnInputs) {
        RequestState state = requestStates.computeIfAbsent(request, ignored -> new RequestState());
        if (state.inputCount == turnInputs.size() && !state.hadToolCalls) return List.of();
        state.inputCount = turnInputs.size();
        String apiKey = SecretsUtils.getSecretText(configuration.getApiKeyCredentialsId(), null);
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Gemini API key credential could not be resolved for model configuration: "
                    + configuration.getModelId());
        }

        StringBuilder systemText = new StringBuilder();
        List<Content> contents = new ArrayList<>();
        List<Part> currentUserParts = new ArrayList<>();
        GenerateContentConfig.Builder config = GenerateContentConfig.builder();
        List<Tool> tools = new ArrayList<>();
        int maxTokens = clientConfiguration == null
                ? (int) GeminiClientSettings.DEFAULT_MAX_TOKENS
                : (int) clientConfiguration.getDefaultMaxTokens();
        config.maxOutputTokens(maxTokens);

        for (ModelInput input : turnInputs) {
            switch (input) {
                case SystemPromptContent value -> append(systemText, value.getSystemPrompt());
                case InputSkillContent value ->
                    append(systemText, value.getSkill().getSkillText());
                case InputTextContent value -> currentUserParts.add(Part.fromText(value.getText()));
                case InputImageContent value ->
                    currentUserParts.add(
                            Part.fromBytes(Base64.getDecoder().decode(value.getImageBase64()), "image/png"));
                case InputFileContent value ->
                    currentUserParts.add(Part.fromBytes(value.getFileData(), value.getContentType()));
                case InputVideoContent value -> currentUserParts.add(Part.fromBytes(value.getVideoData(), "video/mp4"));
                case InputAudioContent value ->
                    currentUserParts.add(Part.fromBytes(value.getAudioData(), "audio/mpeg"));
                case ToolCallResponseContent value -> {
                    flushUser(contents, currentUserParts);
                    if (state.previousModelContent != null
                            && contents.stream().noneMatch(content -> content == state.previousModelContent)) {
                        contents.add(state.previousModelContent);
                    }
                    String name = value.getCalledTool() == null
                            ? "unknown"
                            : value.getCalledTool().getName();
                    Map<String, Object> response = Map.of(
                            "result",
                            value.getResponseContent() == null ? "" : value.getResponseContent(),
                            "successful",
                            value.isSuccessful());
                    contents.add(Content.builder()
                            .role("user")
                            .parts(Part.fromFunctionResponse(name, response))
                            .build());
                }
                case InputToolContent value ->
                    tools.add(
                            Tool.builder().functionDeclarations(function(value)).build());
                case InputConversationContent value -> appendConversation(contents, value);
                case MaxOutputTokensContent value ->
                    config.maxOutputTokens(Math.toIntExact(value.getMaxOutputTokens()));
                case TemperatureContent value ->
                    config.temperature((float) validateRange("temperature", value.getTemperature(), 0, 2));
                case TopPContent value -> config.topP((float) validateRange("top_p", value.getTopP(), 0, 1));
                case StopSequencesContent value -> config.stopSequences(value.getStopPhrases());
                case ThinkingLevelContent value -> configureThinking(config, value.getThinkingLevel());
                default -> {}
            }
        }
        flushUser(contents, currentUserParts);
        if (contents.isEmpty()) contents.add(Content.fromParts(Part.fromText("")));
        if (!systemText.isEmpty()) config.systemInstruction(Content.fromParts(Part.fromText(systemText.toString())));
        if (!tools.isEmpty()) config.tools(tools);

        int timeout = clientConfiguration == null || clientConfiguration.getTimeoutSeconds() <= 0
                ? 120_000
                : Math.toIntExact(clientConfiguration.getTimeoutSeconds() * 1000);
        try (Client client = Client.builder()
                .apiKey(apiKey)
                .httpOptions(HttpOptions.builder().timeout(timeout).build())
                .build()) {
            GenerateContentResponse response =
                    client.models.generateContent(configuration.getModelName(), contents, config.build());
            List<ModelOutput> outputs = new ArrayList<>();
            boolean toolCall = false;
            List<Part> responseParts = response.parts();
            if (responseParts == null) {
                throw new IllegalStateException("Gemini API response did not contain response parts");
            }
            for (Part part : responseParts) {
                if (part.text().isPresent()) {
                    if (part.thought().orElse(false))
                        outputs.add(new ThinkingContent(part.text().get()));
                    else outputs.add(new OutputTextContent(part.text().get()));
                }
                part.functionCall().ifPresent(call -> {
                    JsonObject args =
                            GSON.toJsonTree(call.args().orElse(Map.of())).getAsJsonObject();
                    outputs.add(new ToolCallContent(
                            call.id().orElse(UUID.randomUUID().toString()),
                            args,
                            call.name().orElseThrow()));
                });
                part.inlineData()
                        .ifPresent(blob -> blob.data().ifPresent(data -> {
                            if (blob.mimeType().orElse("").startsWith("audio/"))
                                outputs.add(new OutputAudioContent(data));
                        }));
                if (part.functionCall().isPresent()) toolCall = true;
            }
            state.previousModelContent = response.candidates()
                    .flatMap(candidates -> candidates.stream().findFirst())
                    .flatMap(candidate -> candidate.content())
                    .orElse(null);
            response.usageMetadata()
                    .ifPresent(usage -> outputs.add(new TokenUtilizationContent(
                            usage.promptTokenCount().orElse(0),
                            usage.candidatesTokenCount().orElse(0),
                            usage.cachedContentTokenCount().orElse(0))));
            boolean responseHadToolCalls = toolCall;
            response.candidates()
                    .flatMap(candidates -> candidates.stream().findFirst())
                    .flatMap(candidate -> candidate.finishReason())
                    .ifPresent(reason -> {
                        if (reason.toString().contains("MAX_TOKENS"))
                            outputs.add(new FinishReasonContent(ModelFinishReason.TOKEN_CAP));
                        else if (reason.toString().contains("SAFETY"))
                            outputs.add(new FinishReasonContent(ModelFinishReason.SAFEGUARD));
                        else if (!responseHadToolCalls) outputs.add(new FinishReasonContent(ModelFinishReason.STOP));
                    });
            state.hadToolCalls = toolCall;
            return filterOutputs(request, outputs);
        } catch (RuntimeException e) {
            state.hadToolCalls = false;
            throw new IllegalStateException("Gemini API request failed: " + e.getMessage(), e);
        }
    }

    private static void flushUser(List<Content> contents, List<Part> parts) {
        if (!parts.isEmpty()) {
            contents.add(
                    Content.builder().role("user").parts(List.copyOf(parts)).build());
            parts.clear();
        }
    }

    private static void append(StringBuilder target, String text) {
        if (text != null && !text.isBlank()) {
            if (!target.isEmpty()) target.append("\n\n");
            target.append(text);
        }
    }

    private static void appendConversation(List<Content> contents, InputConversationContent input) {
        if (input.getConversation() == null) return;
        List<Part> parts = new ArrayList<>();
        input.getConversation().getConversation().forEach(content -> {
            if (content instanceof InputTextContent text) parts.add(Part.fromText(text.getText()));
            else if (content instanceof OutputTextContent text)
                contents.add(Content.builder()
                        .role("model")
                        .parts(Part.fromText(text.getText()))
                        .build());
        });
        if (!parts.isEmpty())
            contents.add(Content.builder().role("user").parts(parts).build());
    }

    private static FunctionDeclaration function(InputToolContent input) {
        Map<String, Schema> properties = new LinkedHashMap<>();
        List<String> required = new ArrayList<>();
        for (ToolArgumentDescription argument : input.getTool().getArguments()) {
            properties.put(
                    argument.getName(),
                    Schema.builder()
                            .type(schemaType(argument.getType()))
                            .description(argument.getDescription())
                            .build());
            if (argument.isRequired()) required.add(argument.getName());
        }
        return FunctionDeclaration.builder()
                .name(input.getTool().getName())
                .description(input.getTool().getDescription())
                .parameters(Schema.builder()
                        .type("OBJECT")
                        .properties(properties)
                        .required(required)
                        .build())
                .build();
    }

    private static String schemaType(Class<?> type) {
        if (type == String.class || type == Character.class || type == char.class) return "STRING";
        if (type == boolean.class || type == Boolean.class) return "BOOLEAN";
        if (Number.class.isAssignableFrom(type) || type.isPrimitive()) return "NUMBER";
        if (type.isArray() || Collection.class.isAssignableFrom(type)) return "ARRAY";
        return "OBJECT";
    }

    private static void configureThinking(GenerateContentConfig.Builder config, ModelThinkingLevel level) {
        if (level != ModelThinkingLevel.OFF)
            config.thinkingConfig(ThinkingConfig.builder()
                    .includeThoughts(true)
                    .thinkingLevel(level.name())
                    .build());
    }

    private static double validateRange(String name, double value, double min, double max) {
        if (value < min || value > max)
            throw new IllegalArgumentException(name + " must be between " + min + " and " + max);
        return value;
    }

    private static List<ModelOutput> filterOutputs(ModelRequest request, List<ModelOutput> outputs) {
        Set<Class<ModelOutput>> requested = request.getOutputClasses();
        if (requested == null || requested.isEmpty()) return outputs;
        return outputs.stream()
                .filter(output ->
                        output instanceof ToolCallContent || requested.stream().anyMatch(c -> c.isInstance(output)))
                .toList();
    }

    @Extension
    public static class DescriptorImpl extends Descriptor<ModelClient<?, ?>> {
        @Override
        public @NonNull String getDisplayName() {
            return "Gemini Client";
        }
    }
}
