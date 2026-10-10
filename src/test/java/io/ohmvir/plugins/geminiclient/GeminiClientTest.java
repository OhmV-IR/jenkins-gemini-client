package io.ohmvir.plugins.geminiclient;

import static org.junit.jupiter.api.Assertions.*;

import com.google.genai.types.HttpOptions;
import com.google.genai.types.Model;
import com.google.genai.types.ThinkingConfig;
import com.google.genai.types.ThinkingLevel;
import hudson.model.Descriptor;
import hudson.util.FormValidation;
import hudson.util.ListBoxModel;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.ModelCapability;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.ModelData;
import io.ohmvir.plugins.jenkinsaisynapse.api.models.ModelThinkingLevel;
import java.io.IOException;
import java.util.List;
import net.sf.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.kohsuke.stapler.StaplerRequest2;

public class GeminiClientTest {
    @Test
    public void testClientSettingsDefaults() throws Descriptor.FormException {
        GeminiClientSettings settings = new GeminiClientSettings();
        assertEquals(4096L, settings.getDefaultMaxTokens());
        assertEquals(120L, settings.getTimeoutSeconds());
        assertEquals("Gemini Client Settings", settings.getDisplayName());
    }

    @Test
    public void testClientSettingsCustom() throws Descriptor.FormException {
        GeminiClientSettings settings = new GeminiClientSettings(60L, 8192L);
        assertEquals(8192L, settings.getDefaultMaxTokens());
        assertEquals(60L, settings.getTimeoutSeconds());
        assertThrows(Descriptor.FormException.class, () -> new GeminiClientSettings(60L, 0L));
    }

    @Test
    public void testModelSettingsValidation() {
        GeminiModelSettings.DescriptorImpl desc = new GeminiModelSettings.DescriptorImpl();
        assertEquals(FormValidation.Kind.ERROR, desc.doCheckApiKeyCredentialsId("").kind);
        assertEquals(FormValidation.Kind.ERROR, desc.doCheckApiKeyCredentialsId(null).kind);
        assertEquals(FormValidation.Kind.OK, desc.doCheckApiKeyCredentialsId("dummy-id-outside-jenkins").kind);
        assertEquals(FormValidation.Kind.ERROR, desc.doCheckModelName("").kind);
        assertEquals(FormValidation.Kind.ERROR, desc.doCheckModelName(null).kind);
        assertEquals(FormValidation.Kind.OK, desc.doCheckModelName("gemini-2.5-flash").kind);
    }

    @Test
    public void testModelSettingsConstructorValidation() {
        assertThrows(Descriptor.FormException.class, () -> new GeminiModelSettings("", "valid-id"));
        assertThrows(Descriptor.FormException.class, () -> new GeminiModelSettings("gemini-2.5-flash", ""));
        assertThrows(Descriptor.FormException.class, () -> new GeminiModelSettings("gemini-2.5-flash", null));
    }

    @Test
    public void testModelSettingsDropdownNoFallbackWhenNoCredentials() {
        GeminiModelSettings.DescriptorImpl desc = new GeminiModelSettings.DescriptorImpl();
        ListBoxModel items = desc.doFillModelNameItems("");
        assertEquals(1, items.size());
        assertTrue(items.get(0).name.contains("Please select a valid API Key credential"));
    }

    @Test
    public void testModelDataRetrieverThrowsWhenCredentialsUnresolvable() {
        GeminiModelDataRetriever retriever = new GeminiModelDataRetriever();
        GeminiModelSettings config;
        try {
            config = new GeminiModelSettings();
        } catch (Descriptor.FormException e) {
            fail("Default config should not throw: " + e.getMessage());
            return;
        }
        assertThrows(IOException.class, () -> retriever.retrieveFromConfiguration(config));
    }

    @Test
    public void testClientSettingsSetterAndConfigureValidation() throws Descriptor.FormException {
        GeminiClientSettings settings = new GeminiClientSettings();
        settings.setDefaultMaxTokens(2048L);
        assertEquals(2048L, settings.getDefaultMaxTokens());
        assertThrows(Descriptor.FormException.class, () -> settings.setDefaultMaxTokens(-1L));
        assertThrows(
                Descriptor.FormException.class,
                () -> settings.configure((StaplerRequest2) null, JSONObject.fromObject("{\"defaultMaxTokens\": 0}")));
        assertThrows(
                Descriptor.FormException.class,
                () -> settings.configure((StaplerRequest2) null, JSONObject.fromObject("{\"timeoutSeconds\": -5}")));
        assertEquals(2048L, settings.getDefaultMaxTokens(), "Rejected submissions must not change settings");
    }

    @Test
    public void testHttpTimeouts() throws Descriptor.FormException {
        assertEquals(
                GeminiClient.DEFAULT_TIMEOUT_MILLIS,
                GeminiClient.httpOptions(null).timeout().orElseThrow());
        assertEquals(
                30_000,
                GeminiClient.httpOptions(new GeminiClientSettings(30L, 100L))
                        .timeout()
                        .orElseThrow());
        assertTrue(
                GeminiClient.httpOptions(new GeminiClientSettings(0L, 100L))
                        .timeout()
                        .isEmpty(),
                "A timeout of 0 means no timeout");
        HttpOptions huge = GeminiClient.httpOptions(new GeminiClientSettings(Long.MAX_VALUE, 100L));
        assertTrue(huge.timeout().orElseThrow() > 0);
    }

    @Test
    public void testThinkingConfigForGemini25UsesBudgets() {
        assertEquals(
                0,
                GeminiClient.thinkingConfig("gemini-2.5-flash", ModelThinkingLevel.OFF)
                        .thinkingBudget()
                        .orElseThrow());
        ThinkingConfig low = GeminiClient.thinkingConfig("models/gemini-2.5-pro", ModelThinkingLevel.LOW);
        assertEquals(1024, low.thinkingBudget().orElseThrow());
        assertTrue(low.includeThoughts().orElseThrow());
        assertTrue(low.thinkingLevel().isEmpty());
        assertEquals(
                24576,
                GeminiClient.thinkingConfig("gemini-2.5-flash", ModelThinkingLevel.MAX)
                        .thinkingBudget()
                        .orElseThrow());
    }

    @Test
    public void testThinkingConfigForNewerModelsUsesSupportedLevels() {
        assertEquals(
                ThinkingLevel.Known.MINIMAL,
                GeminiClient.thinkingConfig("gemini-3-flash", ModelThinkingLevel.OFF)
                        .thinkingLevel()
                        .orElseThrow()
                        .knownEnum());
        for (ModelThinkingLevel level :
                List.of(ModelThinkingLevel.HIGH, ModelThinkingLevel.EXTRA_HIGH, ModelThinkingLevel.MAX)) {
            ThinkingConfig config = GeminiClient.thinkingConfig("gemini-3-pro-preview", level);
            assertEquals(
                    ThinkingLevel.Known.HIGH,
                    config.thinkingLevel().orElseThrow().knownEnum(),
                    level.name());
            assertTrue(config.thinkingBudget().isEmpty());
        }
        ThinkingConfig on = GeminiClient.thinkingConfig("gemini-3-pro-preview", ModelThinkingLevel.ON);
        assertTrue(on.includeThoughts().orElseThrow());
        assertTrue(on.thinkingLevel().isEmpty());
    }

    @Test
    public void testModelDataForThinkingModel() throws Descriptor.FormException {
        Model model = Model.fromJson("""
                {"name": "models/gemini-2.5-flash", "inputTokenLimit": 1048576, "outputTokenLimit": 65536,
                 "thinking": true}
                """);
        ModelData data =
                GeminiModelDataRetriever.toModelData(new GeminiModelSettings("gemini-2.5-flash", "key"), model);
        assertEquals("gemini:gemini-2.5-flash", data.getModelId());
        assertTrue(data.hasCapability(ModelCapability.THINKING));
        assertTrue(data.supportsThinkingLevel(ModelThinkingLevel.OFF));
        assertTrue(data.supportsThinkingLevel(ModelThinkingLevel.ON));
        assertEquals(1048576L, data.getContextWindow());
        assertEquals(1048576L, data.getMaxInputTokens());
        assertEquals(65536L, data.getMaxOutputTokens());
        assertEquals(2.0d, data.getMaxTemperature());
    }

    @Test
    public void testModelDataForProModelCannotDisableThinking() throws Descriptor.FormException {
        Model model = Model.fromJson("{\"name\": \"models/gemini-2.5-pro\", \"thinking\": true}");
        ModelData data = GeminiModelDataRetriever.toModelData(new GeminiModelSettings("gemini-2.5-pro", "key"), model);
        assertFalse(data.supportsThinkingLevel(ModelThinkingLevel.OFF));
        assertTrue(data.supportsThinkingLevel(ModelThinkingLevel.HIGH));
    }

    @Test
    public void testModelDataForNonThinkingModel() throws Descriptor.FormException {
        Model model = Model.fromJson("{\"name\": \"models/gemini-2.0-flash\", \"maxTemperature\": 1.5}");
        ModelData data =
                GeminiModelDataRetriever.toModelData(new GeminiModelSettings("gemini-2.0-flash", "key"), model);
        assertFalse(data.hasCapability(ModelCapability.THINKING));
        assertFalse(data.supportsThinking());
        assertTrue(data.hasCapability(ModelCapability.TOOLS));
        assertEquals(1.5d, data.getMaxTemperature());
        assertEquals(0L, data.getContextWindow());
        assertNull(data.getMaxOutputTokens());
    }

    @Test
    public void testModelDropdownOnlyListsGenerateContentModels() {
        assertTrue(GeminiModelSettings.DescriptorImpl.supportsGenerateContent(
                Model.fromJson("{\"supportedActions\": [\"generateContent\", \"countTokens\"]}")));
        assertFalse(GeminiModelSettings.DescriptorImpl.supportsGenerateContent(
                Model.fromJson("{\"supportedActions\": [\"embedContent\"]}")));
        assertTrue(GeminiModelSettings.DescriptorImpl.supportsGenerateContent(Model.fromJson("{}")));
    }
}
