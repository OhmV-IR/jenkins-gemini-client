package io.ohmvir.plugins.geminiclient;

import com.google.genai.Client;
import com.google.genai.types.HttpOptions;
import com.google.genai.types.Model;
import hudson.Extension;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.*;
import io.ohmvir.plugins.jenkinsaisynapse.utils.SecretsUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Extension
public class GeminiModelDataRetriever extends ModelDataRetriever<GeminiModelSettings> {
    public GeminiModelDataRetriever() {
        super(GeminiModelSettings.class);
    }

    @Override
    public ModelData retrieveFromConfiguration(GeminiModelSettings configuration)
            throws IOException, InterruptedException {
        String apiKey = SecretsUtils.getSecretText(configuration.getApiKeyCredentialsId(), null);
        if (apiKey == null || apiKey.isBlank()) {
            throw new IOException("Gemini API key credential could not be resolved for model configuration: "
                    + configuration.getModelId());
        }
        String modelName = configuration.getModelName();
        if (modelName == null || modelName.isBlank())
            throw new IOException("Model name is required for model configuration: " + configuration.getModelId());
        try (Client client = Client.builder()
                .apiKey(apiKey)
                .httpOptions(HttpOptions.builder().timeout(15_000).build())
                .build()) {
            Model model = client.models.get(modelName, null);
            ModelData data = new ModelData(configuration);
            List<ModelCapability> capabilities = new ArrayList<>(List.of(
                    ModelCapability.TOOLS,
                    ModelCapability.SKILLS,
                    ModelCapability.ADJUSTABLE_SYSTEM_PROMPT,
                    ModelCapability.CONVERSATIONS,
                    ModelCapability.OUTPUT_TOKEN_LIMITING,
                    ModelCapability.PREMATURE_STOP,
                    ModelCapability.CUSTOM_STOP_SEQUENCES,
                    ModelCapability.CUSTOM_TEMPERATURE,
                    ModelCapability.CUSTOM_TOP_P,
                    ModelCapability.THINKING,
                    ModelCapability.TOKEN_USAGE_METRICS));
            data.setCapabilities(capabilities);
            data.setSupportedThinkingLevels(List.of(
                    ModelThinkingLevel.OFF,
                    ModelThinkingLevel.LOW,
                    ModelThinkingLevel.MEDIUM,
                    ModelThinkingLevel.HIGH,
                    ModelThinkingLevel.EXTRA_HIGH,
                    ModelThinkingLevel.MAX));
            data.setInputs(List.of(
                    ModelInputType.TEXT,
                    ModelInputType.IMAGE,
                    ModelInputType.FILE,
                    ModelInputType.VIDEO,
                    ModelInputType.AUDIO));
            data.setOutputs(List.of(ModelOutputType.UNSTRUCTURED_TEXT, ModelOutputType.STRUCTURED_OUTPUT));
            data.setContextWindow(
                    model.inputTokenLimit().map(Integer::longValue).orElse(0L));
            data.setMaxInputTokens(
                    model.inputTokenLimit().map(Integer::longValue).orElse(null));
            data.setMaxOutputTokens(
                    model.outputTokenLimit().map(Integer::longValue).orElse(null));
            data.setMaxTemperature(
                    model.maxTemperature().map(Float::doubleValue).orElse(2.0d));
            return data;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed to retrieve model info for '" + modelName + "' from Gemini API: " + e.getMessage(), e);
        }
    }
}
