package io.ohmvir.plugins.geminiclient;

import hudson.Extension;
import io.ohmvir.plugins.jenkinsaisynapse.configuration.client.ModelClientConfiguration;
import jenkins.model.Jenkins;
import lombok.Getter;
import net.sf.json.JSONObject;
import org.jspecify.annotations.NonNull;
import org.kohsuke.stapler.DataBoundConstructor;
import org.kohsuke.stapler.DataBoundSetter;
import org.kohsuke.stapler.StaplerRequest2;

@Extension
public class GeminiClientSettings extends ModelClientConfiguration {
    public static final long DEFAULT_MAX_TOKENS = 4096L;

    private @Getter long defaultMaxTokens;

    public GeminiClientSettings() throws FormException {
        super(120L);
        this.defaultMaxTokens = DEFAULT_MAX_TOKENS;
        if (Jenkins.getInstanceOrNull() != null) load();
    }

    @DataBoundConstructor
    public GeminiClientSettings(long timeoutSeconds, long defaultMaxTokens) throws FormException {
        super(timeoutSeconds);
        this.defaultMaxTokens = validateDefaultMaxTokens(defaultMaxTokens);
    }

    @DataBoundSetter
    public void setDefaultMaxTokens(long defaultMaxTokens) throws FormException {
        this.defaultMaxTokens = validateDefaultMaxTokens(defaultMaxTokens);
    }

    private static long validateDefaultMaxTokens(long defaultMaxTokens) throws FormException {
        if (defaultMaxTokens <= 0) {
            throw new FormException("Default max tokens must be greater than zero", "defaultMaxTokens");
        }
        return defaultMaxTokens;
    }

    @Override
    public boolean configure(StaplerRequest2 req, JSONObject json) throws FormException {
        // Global configuration is bound onto this instance through setters, so validate before binding.
        if (json.optLong("timeoutSeconds", 0L) < 0) {
            throw new FormException("Timeout seconds must not be negative", "timeoutSeconds");
        }
        validateDefaultMaxTokens(json.optLong("defaultMaxTokens", DEFAULT_MAX_TOKENS));
        req.bindJSON(this, json);
        save();
        return true;
    }

    @Override
    public @NonNull String getDisplayName() {
        return "Gemini Client Settings";
    }
}
