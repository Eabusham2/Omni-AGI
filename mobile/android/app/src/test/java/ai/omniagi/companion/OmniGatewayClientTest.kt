package ai.omniagi.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class OmniGatewayClientTest {
    @Test
    fun allowsHttpOnlyForLoopbackOrEmulatorAndRequiresHttpsPin() {
        val pin = "AB".repeat(32)
        assertEquals(
            "http://10.0.2.2:43123",
            OmniGatewayClient.normalizeEndpoint("  http://10.0.2.2:43123/  "),
        )
        assertEquals(
            "https://192.168.1.20:41837/#sha256=${pin.lowercase()}",
            OmniGatewayClient.normalizeEndpoint("https://192.168.1.20:41837/#sha256=$pin"),
        )
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("file:///tmp/brain")
        }
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("http://user:secret@10.0.2.2:43123")
        }
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("http://10.0.2.2:43123/v1/brains")
        }
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("http://example.com:41837")
        }
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("http://192.168.1.20:41837")
        }
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("https://192.168.1.20:41837")
        }
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("http://10.0.2.2:43123?token=wrong-place")
        }
        assertThrows(IllegalArgumentException::class.java) {
            OmniGatewayClient.normalizeEndpoint("http://10.0.2.2:43123#fragment")
        }
    }
}
