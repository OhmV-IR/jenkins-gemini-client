package io.ohmvir.plugins.geminiclient;

import com.cloudbees.plugins.credentials.CredentialsMatchers;
import com.cloudbees.plugins.credentials.common.StandardCredentials;
import com.cloudbees.plugins.credentials.common.StandardListBoxModel;
import com.google.genai.Client;
import com.google.genai.types.HttpOptions;
import com.google.genai.types.ListModelsConfig;
import com.google.genai.types.Model;
import hudson.Extension;
import hudson.model.Descriptor;
import hudson.model.Item;
import hudson.security.ACL;
import hudson.util.FormValidation;
import hudson.util.ListBoxModel;
import io.ohmvir.plugins.jenkinsaisynapse.configuration.models.ModelConfiguration;
import io.ohmvir.plugins.jenkinsaisynapse.utils.SecretsUtils;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;
import jenkins.model.Jenkins;
import lombok.Getter;
import org.jenkinsci.plugins.plaincredentials.StringCredentials;
import org.jspecify.annotations.NonNull;
import org.kohsuke.stapler.AncestorInPath;
import org.kohsuke.stapler.DataBoundConstructor;
import org.kohsuke.stapler.QueryParameter;
import org.kohsuke.stapler.verb.POST;

@Extension
public class GeminiModelSettings extends ModelConfiguration {
    public static final String DEFAULT_MODEL = "gemini-2.5-flash";
    private static final Logger LOGGER = Logger.getLogger(GeminiModelSettings.class.getName());
    private final @Getter String apiKeyCredentialsId;

    public GeminiModelSettings() throws Descriptor.FormException {
        super(DEFAULT_MODEL, "Gemini 2.5 Flash");
        this.apiKeyCredentialsId = null;
    }

    @DataBoundConstructor
    public GeminiModelSettings(String modelName, String apiKeyCredentialsId) throws Descriptor.FormException {
        super(validateModelName(modelName), validateModelName(modelName));
        if (apiKeyCredentialsId == null || apiKeyCredentialsId.isBlank()) {
            throw new Descriptor.FormException("API Key Credential is required", "apiKeyCredentialsId");
        }
        if (Jenkins.getInstanceOrNull() != null && SecretsUtils.getSecretText(apiKeyCredentialsId, null) == null) {
            throw new Descriptor.FormException(
                    "apiKeyCredentialsId does not resolve to a valid string credential", "apiKeyCredentialsId");
        }
        this.apiKeyCredentialsId = apiKeyCredentialsId;
    }

    private static String validateModelName(String modelName) throws Descriptor.FormException {
        if (modelName == null || modelName.isBlank()) {
            throw new Descriptor.FormException("Model name is required and cannot be empty", "modelName");
        }
        return modelName.trim();
    }

    @Override
    public String getProviderType() {
        return "gemini";
    }

    @Extension
    public static class DescriptorImpl extends ModelConfiguration.DescriptorImpl {
        @Override
        public @NonNull String getDisplayName() {
            return "Gemini Model";
        }

        public ListBoxModel doFillModelNameItems(@QueryParameter String apiKeyCredentialsId) {
            ListBoxModel models = new ListBoxModel();
            if (apiKeyCredentialsId == null || apiKeyCredentialsId.isBlank()) {
                models.add(new ListBoxModel.Option(
                        "Please select a valid API Key credential to populate models", "", true));
                return models;
            }
            String apiKey =
                    Jenkins.getInstanceOrNull() == null ? null : SecretsUtils.getSecretText(apiKeyCredentialsId, null);
            if (apiKey == null || apiKey.isBlank()) {
                models.add(new ListBoxModel.Option(
                        "Please select a valid API Key credential to populate models", "", true));
                return models;
            }
            try (Client client = Client.builder()
                    .apiKey(apiKey)
                    .httpOptions(HttpOptions.builder().timeout(10_000).build())
                    .build()) {
                for (Model model : client.models.list(ListModelsConfig.builder().build())) {
                    String name = model.name().orElse("");
                    if (name.startsWith("models/")) name = name.substring("models/".length());
                    if (!name.isBlank()) models.add(name, name);
                }
            } catch (Exception e) {
                LOGGER.log(Level.WARNING, "Failed to retrieve available models from Gemini API", e);
                models.add(new ListBoxModel.Option("Error retrieving models from Gemini: " + e.getMessage(), "", true));
            }
            return models;
        }

        public ListBoxModel doFillApiKeyCredentialsIdItems(
                @AncestorInPath Item context, @QueryParameter String apiKeyCredentialsId) {
            if (Jenkins.getInstanceOrNull() == null)
                return new StandardListBoxModel().includeCurrentValue(apiKeyCredentialsId);
            if (context == null
                    ? !Jenkins.get().hasPermission(Jenkins.ADMINISTER)
                    : !context.hasPermission(Item.CONFIGURE)) {
                return new StandardListBoxModel().includeCurrentValue(apiKeyCredentialsId);
            }
            return new StandardListBoxModel()
                    .includeEmptyValue()
                    .includeMatchingAs(
                            ACL.SYSTEM2,
                            context,
                            StandardCredentials.class,
                            Collections.emptyList(),
                            CredentialsMatchers.instanceOf(StringCredentials.class));
        }

        @POST
        public FormValidation doCheckApiKeyCredentialsId(@QueryParameter String value) {
            if (value == null || value.trim().isEmpty()) return FormValidation.error("API Key Credential is required");
            if (Jenkins.getInstanceOrNull() != null && SecretsUtils.getSecretText(value, null) == null) {
                return FormValidation.error(
                        "The selected credential ID could not be found or does not resolve to a valid string credential");
            }
            return FormValidation.ok();
        }

        @POST
        @Override
        public FormValidation doCheckModelName(@QueryParameter String value) {
            return value == null || value.trim().isEmpty()
                    ? FormValidation.error("Model name is required and cannot be empty")
                    : FormValidation.ok();
        }
    }
}
