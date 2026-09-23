package io.ohmvir.plugins.geminiclient;

import static org.junit.jupiter.api.Assertions.*;

import hudson.model.Descriptor;
import hudson.util.FormValidation;
import hudson.util.ListBoxModel;
import java.io.IOException;
import org.junit.jupiter.api.Test;

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
}
