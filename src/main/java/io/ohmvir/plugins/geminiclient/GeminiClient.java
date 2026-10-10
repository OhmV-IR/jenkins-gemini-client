package io.ohmvir.plugins.geminiclient;

import com.google.genai.Client;
import com.google.genai.types.Candidate;
import com.google.genai.types.Content;
import com.google.genai.types.FinishReason;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.HttpOptions;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import com.google.genai.types.ThinkingConfig;
import com.google.genai.types.ThinkingLevel;
import com.google.genai.types.Tool;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import hudson.Extension;
import hudson.model.Descriptor;
import io.ohmvir.plugins.jenkinsaisynapse.api.ModelContent;
import io.ohmvir.plugins.jenkinsaisynapse.api.client.ModelClient;
import io.ohmvir.plugins.jenkinsaisynapse.api.input.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.output.*;
import io.ohmvir.plugins.jenkinsaisynapse.api.tools.ToolArgumentDescription;
import io.ohmvir.plugins.jenkinsaisynapse.utils.SecretsUtils;
import java.math.BigInteger;
import java.util.*;
import org.jspecify.annotations.NonNull;

@Extension
public class GeminiClient extends ModelClient<GeminiModelSettings, GeminiClientSettings> {
    private static final Gson GSON = new Gson();
    static final int DEFAULT_TIMEOUT_MILLIS = 120_000;

    private static final Set<FinishReason.Known> SAFEGUARD_FINISH_REASONS = EnumSet.of(
            FinishReason.Known.SAFETY,
            FinishReason.Known.RECITATION,
            FinishReason.Known.BLOCKLIST,
            FinishReason.Known.PROHIBITED_CONTENT,
            FinishReason.Known.SPII,
            FinishReason.Known.IMAGE_SAFETY,
            FinishReason.Known.IMAGE_PROHIBITED_CONTENT,
            FinishReason.Known.IMAGE_RECITATION);

    static final class RequestState {
        /** Number of turn inputs answered by the last successful API call. */
        int inputCount = -1;
        /**
         * Model turns keyed by the number of turn inputs that preceded them. Synapse only feeds tool responses back
         * as inputs, so the model's own turns (function calls, thought signatures) have to be replayed from here.
         */
        final NavigableMap<Integer, Content> modelTurns = new TreeMap<>();
        /** Function calls keyed by the tool use id handed to Synapse. */
        final Map<String, FunctionCall> functionCalls = new HashMap<>();
    }

    private enum TurnKind {
        USER("user"),
        FUNCTION_RESPONSE("user"),
        MODEL("model");

        final String role;

        TurnKind(String role) {
            this.role = role;
        }
    }

    /** Groups consecutive parts of the same kind into a single {@link Content}. */
    private static final class ContentsBuilder {
        final List<Content> contents = new ArrayList<>();
        private final List<Part> parts = new ArrayList<>();
        private TurnKind kind;

        void add(TurnKind kind, Part part) {
            if (kind != this.kind) flush();
            this.kind = kind;
            parts.add(part);
        }

        void addTurn(Content content) {
            flush();
            contents.add(content);
        }

        List<Content> build() {
            flush();
            return contents;
        }

        private void flush() {
            if (!parts.isEmpty()) {
                contents.add(Content.builder()
                        .role(kind.role)
                        .parts(List.copyOf(parts))
                        .build());
                parts.clear();
            }
            kind = null;
        }
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
        // Synapse keeps calling until a step returns nothing; with no new inputs there is nothing left to answer.
        if (state.inputCount == turnInputs.size()) return List.of();

        ContentsBuilder history = new ContentsBuilder();
        ContentsBuilder turn = new ContentsBuilder();
        StringBuilder systemText = new StringBuilder();
        GenerateContentConfig.Builder config = GenerateContentConfig.builder();
        Map<String, FunctionDeclaration> functions = new LinkedHashMap<>();
        config.maxOutputTokens(saturatedInt(
                clientConfiguration == null
                        ? GeminiClientSettings.DEFAULT_MAX_TOKENS
                        : clientConfiguration.getDefaultMaxTokens()));

        for (int i = 0; i < turnInputs.size(); i++) {
            Content modelTurn = state.modelTurns.get(i);
            if (modelTurn != null) turn.addTurn(modelTurn);
            switch (turnInputs.get(i)) {
                case SystemPromptContent value -> append(systemText, value.getSystemPrompt());
                case InputSkillContent value ->
                    append(systemText, value.getSkill().getSkillText());
                case InputTextContent value -> turn.add(TurnKind.USER, Part.fromText(value.getText()));
                case InputImageContent value -> turn.add(TurnKind.USER, Part.fromBytes(pngBytes(value), "image/png"));
                case InputFileContent value ->
                    turn.add(TurnKind.USER, Part.fromBytes(value.getFileData(), value.getContentType()));
                case InputVideoContent value ->
                    turn.add(TurnKind.USER, Part.fromBytes(value.getVideoData(), "video/mp4"));
                case InputAudioContent value ->
                    turn.add(TurnKind.USER, Part.fromBytes(value.getAudioData(), "audio/mpeg"));
                case ToolCallResponseContent value ->
                    turn.add(TurnKind.FUNCTION_RESPONSE, functionResponse(value, state));
                case InputToolContent value ->
                    functions.putIfAbsent(value.getTool().getName(), function(value));
                case InputConversationContent value -> appendConversation(history, value);
                case MaxOutputTokensContent value -> config.maxOutputTokens(saturatedInt(value.getMaxOutputTokens()));
                case TemperatureContent value ->
                    config.temperature((float) validateRange("temperature", value.getTemperature(), 0, 2));
                case TopPContent value -> config.topP((float) validateRange("top_p", value.getTopP(), 0, 1));
                case StopSequencesContent value -> config.stopSequences(value.getStopPhrases());
                case ThinkingLevelContent value ->
                    config.thinkingConfig(thinkingConfig(configuration.getModelName(), value.getThinkingLevel()));
                default -> {}
            }
        }
        // Conversation history always precedes the turn being answered.
        List<Content> contents = new ArrayList<>(history.build());
        contents.addAll(turn.build());
        if (contents.isEmpty()) contents.add(Content.fromParts(Part.fromText("")));
        if (!systemText.isEmpty()) config.systemInstruction(Content.fromParts(Part.fromText(systemText.toString())));
        // Gemini expects all function declarations in a single tool.
        if (!functions.isEmpty())
            config.tools(Tool.builder()
                    .functionDeclarations(List.copyOf(functions.values()))
                    .build());

        GenerateContentResponse response;
        try {
            response = generateContent(configuration, clientConfiguration, contents, config.build());
        } catch (RuntimeException e) {
            throw new IllegalStateException("Gemini API request failed: " + e.getMessage(), e);
        }
        List<ModelOutput> outputs = parseResponse(response, state, turnInputs.size());
        state.inputCount = turnInputs.size();
        return filterOutputs(request, outputs);
    }

    /** Performs the API call. Package-private so tests can replace the network round trip. */
    GenerateContentResponse generateContent(
            GeminiModelSettings configuration,
            GeminiClientSettings clientConfiguration,
            List<Content> contents,
            GenerateContentConfig config) {
        String apiKey = SecretsUtils.getSecretText(configuration.getApiKeyCredentialsId(), null);
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Gemini API key credential could not be resolved for model configuration: "
                    + configuration.getModelId());
        }
        try (Client client = Client.builder()
                .apiKey(apiKey)
                .httpOptions(httpOptions(clientConfiguration))
                .build()) {
            return client.models.generateContent(configuration.getModelName(), contents, config);
        }
    }

    static HttpOptions httpOptions(GeminiClientSettings clientConfiguration) {
        HttpOptions.Builder options = HttpOptions.builder();
        Long timeoutSeconds = clientConfiguration == null ? null : clientConfiguration.getTimeoutSeconds();
        if (timeoutSeconds == null)
            return options.timeout(DEFAULT_TIMEOUT_MILLIS).build();
        // A timeout of 0 means wait indefinitely, which the SDK does when no timeout is set.
        if (timeoutSeconds == 0) return options.build();
        return options.timeout(saturatedInt(Math.min(timeoutSeconds, Integer.MAX_VALUE / 1000L) * 1000L))
                .build();
    }

    static List<ModelOutput> parseResponse(GenerateContentResponse response, RequestState state, int inputCount) {
        List<ModelOutput> outputs = new ArrayList<>();
        Optional<Candidate> candidate =
                response.candidates().flatMap(candidates -> candidates.stream().findFirst());
        if (candidate.isEmpty()) {
            if (response.promptFeedback()
                    .flatMap(feedback -> feedback.blockReason())
                    .isEmpty()) {
                throw new IllegalStateException("Gemini API response did not contain any candidates");
            }
            addUsage(response, outputs);
            outputs.add(new FinishReasonContent(ModelFinishReason.SAFEGUARD));
            return outputs;
        }

        // Read parts from the candidate directly: GenerateContentResponse.parts() throws on finish reasons such as
        // SAFETY or RECITATION, which should be reported as finish reasons instead.
        Optional<Content> content = candidate.get().content();
        List<Part> parts = content.flatMap(Content::parts).orElse(List.of());
        boolean toolCall = false;
        for (Part part : parts) {
            part.text().filter(text -> !text.isEmpty()).ifPresent(text -> {
                if (part.thought().orElse(false)) outputs.add(new ThinkingContent(text));
                else outputs.add(new OutputTextContent(text));
            });
            if (part.functionCall().isPresent()) {
                FunctionCall call = part.functionCall().get();
                String toolUseId = call.id()
                        .filter(id -> !id.isBlank())
                        .orElseGet(() -> UUID.randomUUID().toString());
                String name = call.name()
                        .orElseThrow(
                                () -> new IllegalStateException("Gemini API returned a function call without a name"));
                state.functionCalls.put(toolUseId, call);
                JsonObject args = GSON.toJsonTree(call.args().orElse(Map.of())).getAsJsonObject();
                outputs.add(new ToolCallContent(toolUseId, args, name));
                toolCall = true;
            }
            part.inlineData()
                    .ifPresent(blob -> blob.data().ifPresent(data -> {
                        if (blob.mimeType().orElse("").startsWith("audio/")) outputs.add(new OutputAudioContent(data));
                    }));
        }
        if (!parts.isEmpty()) state.modelTurns.put(inputCount, content.get());
        addUsage(response, outputs);
        if (toolCall) outputs.add(new FinishReasonContent(ModelFinishReason.TOOL_CALLS));
        else
            candidate
                    .get()
                    .finishReason()
                    .ifPresent(reason -> outputs.add(new FinishReasonContent(finishReason(reason))));
        return outputs;
    }

    private static void addUsage(GenerateContentResponse response, List<ModelOutput> outputs) {
        response.usageMetadata()
                .ifPresent(usage -> outputs.add(new TokenUtilizationContent(
                        usage.promptTokenCount().orElse(0),
                        // Gemini reports thinking tokens separately, but they are billed as output tokens.
                        (long) usage.candidatesTokenCount().orElse(0)
                                + usage.thoughtsTokenCount().orElse(0),
                        usage.cachedContentTokenCount().orElse(0))));
    }

    static ModelFinishReason finishReason(FinishReason reason) {
        FinishReason.Known known = reason.knownEnum();
        if (known == FinishReason.Known.MAX_TOKENS) return ModelFinishReason.TOKEN_CAP;
        if (SAFEGUARD_FINISH_REASONS.contains(known)) return ModelFinishReason.SAFEGUARD;
        return ModelFinishReason.STOP;
    }

    private static Part functionResponse(ToolCallResponseContent value, RequestState state) {
        // Gemini matches responses to calls by name (and id when it supplied one), so prefer the original call.
        FunctionCall call = state.functionCalls.get(value.getToolUseId());
        String name = call == null ? null : call.name().orElse(null);
        if (name == null)
            name = value.getCalledTool() == null
                    ? "unknown"
                    : value.getCalledTool().getName();
        String responseContent = value.getResponseContent();
        Map<String, Object> response = value.isSuccessful()
                ? Map.of("output", responseContent == null ? "" : responseContent)
                : Map.of("error", responseContent == null ? "Tool call failed" : responseContent);
        FunctionResponse.Builder functionResponse =
                FunctionResponse.builder().name(name).response(response);
        if (call != null) call.id().filter(id -> !id.isBlank()).ifPresent(functionResponse::id);
        return Part.builder().functionResponse(functionResponse.build()).build();
    }

    private static byte[] pngBytes(InputImageContent value) {
        String encoded = value.getImageBase64();
        if (encoded == null) throw new IllegalArgumentException("Image input could not be encoded as PNG");
        // Synapse encodes images as a data URI, which the Base64 decoder does not accept.
        int comma = encoded.indexOf(',');
        if (encoded.startsWith("data:") && comma >= 0) encoded = encoded.substring(comma + 1);
        return Base64.getDecoder().decode(encoded);
    }

    private static void append(StringBuilder target, String text) {
        if (text != null && !text.isBlank()) {
            if (!target.isEmpty()) target.append("\n\n");
            target.append(text);
        }
    }

    private static void appendConversation(ContentsBuilder history, InputConversationContent input) {
        if (input.getConversation() == null) return;
        for (ModelContent content : input.getConversation().getConversation()) {
            if (content instanceof InputTextContent text && text.getText() != null)
                history.add(TurnKind.USER, Part.fromText(text.getText()));
            else if (content instanceof OutputTextContent text && text.getText() != null)
                history.add(TurnKind.MODEL, Part.fromText(text.getText()));
        }
    }

    private static FunctionDeclaration function(InputToolContent input) {
        FunctionDeclaration.Builder function = FunctionDeclaration.builder()
                .name(input.getTool().getName())
                .description(input.getTool().getDescription());
        List<ToolArgumentDescription> arguments = input.getTool().getArguments();
        // Gemini rejects OBJECT schemas without properties, so argument-less tools declare no parameters.
        if (arguments == null || arguments.isEmpty()) return function.build();
        Map<String, Schema> properties = new LinkedHashMap<>();
        List<String> required = new ArrayList<>();
        for (ToolArgumentDescription argument : arguments) {
            properties.put(argument.getName(), schema(argument.getType(), argument.getDescription()));
            if (argument.isRequired()) required.add(argument.getName());
        }
        return function.parameters(Schema.builder()
                        .type("OBJECT")
                        .properties(properties)
                        .required(required)
                        .build())
                .build();
    }

    static Schema schema(Class<?> type, String description) {
        Schema.Builder schema = Schema.builder().type(schemaType(type));
        if (description != null) schema.description(description);
        // Gemini rejects ARRAY schemas without an items schema.
        if (type.isArray()) schema.items(schema(type.getComponentType(), null));
        else if (Collection.class.isAssignableFrom(type))
            schema.items(Schema.builder().type("STRING").build());
        return schema.build();
    }

    static String schemaType(Class<?> type) {
        if (type == boolean.class || type == Boolean.class) return "BOOLEAN";
        if (type == int.class
                || type == long.class
                || type == short.class
                || type == byte.class
                || type == Integer.class
                || type == Long.class
                || type == Short.class
                || type == Byte.class
                || type == BigInteger.class) return "INTEGER";
        if (type == double.class || type == float.class || Number.class.isAssignableFrom(type)) return "NUMBER";
        if (type.isArray() || Collection.class.isAssignableFrom(type)) return "ARRAY";
        // Synapse tools only accept primitives and strings, and Gemini rejects OBJECT schemas without properties.
        return "STRING";
    }

    static ThinkingConfig thinkingConfig(String modelName, ModelThinkingLevel level) {
        ThinkingConfig.Builder thinking = ThinkingConfig.builder();
        // Gemini 2.x models only understand thinking budgets; newer models take a thinking level instead.
        if (usesThinkingBudget(modelName)) {
            return switch (level) {
                case OFF -> thinking.thinkingBudget(0).build();
                case ON -> thinking.includeThoughts(true).build();
                case LOW -> thinking.includeThoughts(true).thinkingBudget(1024).build();
                case MEDIUM ->
                    thinking.includeThoughts(true).thinkingBudget(8192).build();
                case HIGH ->
                    thinking.includeThoughts(true).thinkingBudget(16384).build();
                case EXTRA_HIGH, MAX ->
                    thinking.includeThoughts(true).thinkingBudget(24576).build();
            };
        }
        return switch (level) {
            case OFF -> thinking.thinkingLevel(ThinkingLevel.Known.MINIMAL).build();
            case ON -> thinking.includeThoughts(true).build();
            case LOW ->
                thinking.includeThoughts(true)
                        .thinkingLevel(ThinkingLevel.Known.LOW)
                        .build();
            case MEDIUM ->
                thinking.includeThoughts(true)
                        .thinkingLevel(ThinkingLevel.Known.MEDIUM)
                        .build();
            case HIGH, EXTRA_HIGH, MAX ->
                thinking.includeThoughts(true)
                        .thinkingLevel(ThinkingLevel.Known.HIGH)
                        .build();
        };
    }

    static boolean usesThinkingBudget(String modelName) {
        String name = modelName == null ? "" : modelName.toLowerCase(Locale.ROOT);
        if (name.startsWith("models/")) name = name.substring("models/".length());
        return name.startsWith("gemini-1") || name.startsWith("gemini-2");
    }

    private static double validateRange(String name, double value, double min, double max) {
        if (value < min || value > max)
            throw new IllegalArgumentException(name + " must be between " + min + " and " + max);
        return value;
    }

    private static int saturatedInt(long value) {
        return (int) Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, value));
    }

    private static List<ModelOutput> filterOutputs(ModelRequest request, List<ModelOutput> outputs) {
        Set<Class<? extends ModelOutput>> requested = request.getOutputClasses();
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
