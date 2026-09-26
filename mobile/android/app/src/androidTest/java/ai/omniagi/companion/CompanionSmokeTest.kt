package ai.omniagi.companion

import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.replaceText
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.platform.app.InstrumentationRegistry
import org.hamcrest.Matchers.containsString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayInputStream

@RunWith(AndroidJUnit4::class)
class CompanionSmokeTest {
    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    private val endpoint: String
        get() {
            val port = InstrumentationRegistry.getArguments()
                .getString("omniGatewayPort", "41837")
            return "http://10.0.2.2:$port"
        }

    @Test
    fun pairsWithTheSameBrainAndStreamsAVisibleTurn() {
        onView(withId(R.id.endpointInput)).perform(replaceText(endpoint))
        onView(withId(R.id.codeInput)).perform(replaceText("123456"))
        onView(withId(R.id.pairButton)).perform(click())

        waitFor(12_000) {
            onView(withId(R.id.chatPanel)).check(matches(isDisplayed()))
        }
        onView(withId(R.id.messageInput)).perform(replaceText("Hello from the emulator"))
        onView(withId(R.id.sendButton)).perform(click())
        waitFor(12_000) {
            onView(withText(containsString("Same-brain emulator reply"))).check(matches(isDisplayed()))
        }
        onView(withText(containsString("Hello from the emulator"))).check(matches(isDisplayed()))
    }

    @Test
    fun streamsAnAttachmentWithoutHoldingTheWholePayloadInTheProtocol() {
        val paired = OmniGatewayClient(endpoint).pair("123456", "instrumentation")
        val client = OmniGatewayClient(paired.endpoint, paired.token)
        val brain = client.listBrains().single()
        val bytes = ByteArray(2 * 1024 * 1024) { index -> (index % 251).toByte() }
        var progress = 0L
        val result = client.uploadExperience(
            brainId = brain.id,
            fileName = "emulator-audio.raw",
            mimeType = "audio/x-raw",
            declaredBytes = bytes.size.toLong(),
            input = ByteArrayInputStream(bytes),
        ) { written, _ -> progress = written }
        assertEquals(bytes.size.toLong(), progress)
        assertEquals(bytes.size.toLong(), result.getLong("bytes"))
        assertTrue(result.getJSONArray("results").length() == 1)
    }

    private fun waitFor(timeoutMs: Long, assertion: () -> Unit) {
        val started = System.currentTimeMillis()
        var lastError: Throwable? = null
        while (System.currentTimeMillis() - started < timeoutMs) {
            try {
                InstrumentationRegistry.getInstrumentation().waitForIdleSync()
                assertion()
                return
            } catch (error: Throwable) {
                lastError = error
                Thread.sleep(100)
            }
        }
        throw AssertionError("Timed out waiting for Android companion UI", lastError)
    }
}
