package io.ohmvir.plugins.geminiclient;

import static org.junit.jupiter.api.Assertions.*;

import com.cloudbees.plugins.credentials.CredentialsScope;
import com.cloudbees.plugins.credentials.SystemCredentialsProvider;
import hudson.ExtensionList;
import hudson.model.Descriptor;
import hudson.model.User;
import hudson.security.ACL;
import hudson.security.ACLContext;
import hudson.util.FormValidation;
import hudson.util.ListBoxModel;
import hudson.util.Secret;
import io.ohmvir.plugins.jenkinsaisynapse.api.input.InputTextContent;
import io.ohmvir.plugins.jenkinsaisynapse.api.input.ModelRequest;
import jenkins.model.Jenkins;
import org.htmlunit.html.HtmlForm;
import org.jenkinsci.plugins.plaincredentials.impl.StringCredentialsImpl;
import org.junit.jupiter.api.Test;
import org.jvnet.hudson.test.JenkinsRule;
import org.jvnet.hudson.test.MockAuthorizationStrategy;
import org.jvnet.hudson.test.junit.jupiter.WithJenkins;

@WithJenkins
public class GeminiJenkinsIntegrationTest {
    private static void addApiKey(String id) throws Exception {
        SystemCredentialsProvider.getInstance()
                .getCredentials()
                .add(new StringCredentialsImpl(CredentialsScope.GLOBAL, id, "Gemini key", Secret.fromString("secret")));
        SystemCredentialsProvider.getInstance().save();
    }

    @Test
    public void clientSettingsAreEditableAndPersisted(JenkinsRule j) throws Exception {
        GeminiClientSettings settings = ExtensionList.lookupSingleton(GeminiClientSettings.class);

        HtmlForm form = j.createWebClient().goTo("manage/configure").getFormByName("config");
        form.getInputByName("_.timeoutSeconds").setValue("30");
        form.getInputByName("_.defaultMaxTokens").setValue("8192");
        j.submit(form);

        assertEquals(30L, settings.getTimeoutSeconds());
        assertEquals(8192L, settings.getDefaultMaxTokens());
        GeminiClientSettings reloaded = new GeminiClientSettings();
        assertEquals(30L, reloaded.getTimeoutSeconds());
        assertEquals(8192L, reloaded.getDefaultMaxTokens());
    }

    @Test
    public void modelSettingsRequireResolvableCredential(JenkinsRule j) throws Exception {
        addApiKey("gemini-key");

        GeminiModelSettings settings = new GeminiModelSettings(" gemini-2.5-flash ", "gemini-key");
        assertEquals("gemini-2.5-flash", settings.getModelName());
        assertEquals("gemini:gemini-2.5-flash", settings.getModelId());
        assertThrows(Descriptor.FormException.class, () -> new GeminiModelSettings("gemini-2.5-flash", "missing"));

        GeminiModelSettings.DescriptorImpl descriptor =
                ExtensionList.lookupSingleton(GeminiModelSettings.DescriptorImpl.class);
        assertEquals(FormValidation.Kind.OK, descriptor.doCheckApiKeyCredentialsId("gemini-key").kind);
        assertEquals(FormValidation.Kind.ERROR, descriptor.doCheckApiKeyCredentialsId("missing").kind);
    }

    @Test
    public void credentialListIncludesStringCredentialsForAdministrators(JenkinsRule j) throws Exception {
        addApiKey("gemini-key");
        GeminiModelSettings.DescriptorImpl descriptor =
                ExtensionList.lookupSingleton(GeminiModelSettings.DescriptorImpl.class);

        ListBoxModel items = descriptor.doFillApiKeyCredentialsIdItems(null, "");

        assertTrue(items.stream().anyMatch(option -> option.value.equals("gemini-key")));
    }

    @Test
    public void credentialBackedEndpointsRequireAdministerPermission(JenkinsRule j) throws Exception {
        addApiKey("gemini-key");
        j.jenkins.setSecurityRealm(j.createDummySecurityRealm());
        j.jenkins.setAuthorizationStrategy(
                new MockAuthorizationStrategy().grant(Jenkins.READ).everywhere().to("reader"));
        GeminiModelSettings.DescriptorImpl descriptor =
                ExtensionList.lookupSingleton(GeminiModelSettings.DescriptorImpl.class);

        try (ACLContext ignored = ACL.as2(User.getById("reader", true).impersonate2())) {
            assertEquals(
                    FormValidation.Kind.OK,
                    descriptor.doCheckApiKeyCredentialsId("missing").kind,
                    "Credential existence must not be revealed to non-administrators");
            ListBoxModel models = descriptor.doFillModelNameItems("gemini-key");
            assertEquals(1, models.size());
            assertTrue(models.get(0).name.contains("Please select a valid API Key credential"));
            ListBoxModel credentials = descriptor.doFillApiKeyCredentialsIdItems(null, "gemini-key");
            assertTrue(credentials.stream().noneMatch(option -> option.name.contains("Gemini key")));
        }
    }

    @Test
    public void takeStepFailsClearlyWhenCredentialCannotBeResolved(JenkinsRule j) throws Exception {
        GeminiClient client = ExtensionList.lookupSingleton(GeminiClient.class);
        ModelRequest request = new ModelRequest();
        request.addInput(new InputTextContent("Hello"));

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> client.takeStep(
                        null,
                        new GeminiModelSettings(),
                        new GeminiClientSettings(),
                        request,
                        request.getModelInputs()));

        assertTrue(failure.getMessage().contains("could not be resolved"), failure.getMessage());
    }
}
