package ai.omniagi.companion

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.util.concurrent.atomic.AtomicReference
import javax.net.ssl.SSLContext
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

data class PairedSession(
    val endpoint: String,
    val token: String,
    val deviceId: String,
    val certificateSha256: String?,
)

data class BrainItem(
    val id: String,
    val name: String,
    val readiness: String,
) {
    override fun toString(): String = if (readiness == "ready") name else "$name · $readiness"
}

data class BrainMessage(
    val id: String,
    val role: String,
    val content: String,
    val createdAt: String,
)

sealed interface ChatFrame {
    data class Token(val delta: String) : ChatFrame
    data class State(val state: String, val detail: String?) : ChatFrame
    data class Action(val label: String) : ChatFrame
    data class Preview(val label: String, val progress: Double?) : ChatFrame
    data class Result(val content: String) : ChatFrame
    data class Failure(val message: String, val cancelled: Boolean) : ChatFrame
}

class GatewayException(
    val status: Int,
    message: String,
) : Exception(message)

class OmniGatewayClient(
    endpoint: String,
    private val token: String? = null,
) {
    private val endpointIdentity = parseEndpoint(endpoint)
    val endpoint: String = endpointIdentity.endpoint
    private val certificateSha256: String? = endpointIdentity.certificateSha256
    private val activeChat = AtomicReference<HttpURLConnection?>(null)

    fun pair(code: String, deviceName: String): PairedSession {
        require(code.matches(Regex("^\\d{6}$"))) { "Enter the six-digit pairing code." }
        val body = JSONObject()
            .put("code", code)
            .put("deviceName", deviceName)
        val response = requestJson("/v1/pair", "POST", body, false)
        val responseFingerprint = response.optString("certificateSha256")
            .lowercase()
            .takeIf { it.isNotBlank() }
        if (certificateSha256 != null && responseFingerprint != certificateSha256) {
            throw GatewayException(495, "Studio's pairing identity did not match the pinned TLS certificate.")
        }
        return PairedSession(
            endpoint = endpointIdentity.pairingAddress,
            token = response.getString("token"),
            deviceId = response.getJSONObject("device").getString("id"),
            certificateSha256 = responseFingerprint,
        )
    }

    fun listBrains(): List<BrainItem> {
        val array = requestJson("/v1/brains", "GET", null, true).getJSONArray("brains")
        return buildList {
            for (index in 0 until array.length()) {
                val value = array.getJSONObject(index)
                add(
                    BrainItem(
                        id = value.getString("id"),
                        name = value.getString("name"),
                        readiness = value.getString("readiness"),
                    ),
                )
            }
        }
    }

    fun listMessages(brainId: String): List<BrainMessage> {
        val array = requestJson(
            "/v1/brains/${encodePath(brainId)}/messages",
            "GET",
            null,
            true,
        ).getJSONArray("messages")
        return buildList {
            for (index in 0 until array.length()) {
                val value = array.getJSONObject(index)
                add(
                    BrainMessage(
                        id = value.getString("id"),
                        role = value.getString("role"),
                        content = value.getString("content"),
                        createdAt = value.getString("createdAt"),
                    ),
                )
            }
        }
    }

    fun streamChat(
        brainId: String,
        input: String,
        turnId: String,
        onFrame: (ChatFrame) -> Unit,
    ) {
        val connection = open(
            "/v1/brains/${encodePath(brainId)}/chat",
            "POST",
            true,
        )
        activeChat.set(connection)
        try {
            connection.setRequestProperty("content-type", "application/json; charset=utf-8")
            connection.setRequestProperty("accept", "application/x-ndjson")
            connection.doOutput = true
            val bytes = JSONObject()
                .put("input", input)
                .put("turnId", turnId)
                .toString()
                .toByteArray(StandardCharsets.UTF_8)
            connection.setFixedLengthStreamingMode(bytes.size)
            connection.outputStream.use { it.write(bytes) }
            ensureSuccess(connection)
            BufferedReader(InputStreamReader(connection.inputStream, StandardCharsets.UTF_8)).use { reader ->
                while (true) {
                    val line = reader.readLine() ?: break
                    if (line.isBlank()) continue
                    parseChatFrame(JSONObject(line))?.let(onFrame)
                }
            }
        } finally {
            activeChat.compareAndSet(connection, null)
            connection.disconnect()
        }
    }

    fun disconnectActiveChat() {
        activeChat.getAndSet(null)?.disconnect()
    }

    fun cancel(brainId: String, turnId: String): Int = requestJson(
        "/v1/brains/${encodePath(brainId)}/chat/${encodePath(turnId)}/cancel",
        "POST",
        JSONObject(),
        true,
    ).optInt("cancelled", 0)

    fun uploadExperience(
        brainId: String,
        fileName: String,
        mimeType: String,
        declaredBytes: Long?,
        input: InputStream,
        onProgress: (writtenBytes: Long, totalBytes: Long?) -> Unit,
    ): JSONObject {
        val connection = open(
            "/v1/brains/${encodePath(brainId)}/experience",
            "POST",
            true,
        )
        try {
            connection.setRequestProperty("content-type", mimeType.ifBlank { "application/octet-stream" })
            connection.setRequestProperty(
                "x-omni-filename",
                URLEncoder.encode(fileName, StandardCharsets.UTF_8.name()),
            )
            connection.setRequestProperty("accept", "application/json")
            connection.doOutput = true
            if (declaredBytes != null && declaredBytes >= 0) {
                connection.setFixedLengthStreamingMode(declaredBytes)
            } else {
                connection.setChunkedStreamingMode(256 * 1024)
            }
            BufferedInputStream(input, 256 * 1024).use { source ->
                connection.outputStream.buffered(256 * 1024).use { target ->
                    val buffer = ByteArray(256 * 1024)
                    var written = 0L
                    while (true) {
                        val count = source.read(buffer)
                        if (count < 0) break
                        target.write(buffer, 0, count)
                        written += count
                        onProgress(written, declaredBytes)
                    }
                }
            }
            return readJsonResponse(connection)
        } finally {
            connection.disconnect()
        }
    }

    private fun requestJson(
        path: String,
        method: String,
        body: JSONObject?,
        authorized: Boolean,
    ): JSONObject {
        val connection = open(path, method, authorized)
        try {
            connection.setRequestProperty("accept", "application/json")
            if (body != null) {
                connection.setRequestProperty("content-type", "application/json; charset=utf-8")
                connection.doOutput = true
                val bytes = body.toString().toByteArray(StandardCharsets.UTF_8)
                connection.setFixedLengthStreamingMode(bytes.size)
                connection.outputStream.use { it.write(bytes) }
            }
            return readJsonResponse(connection)
        } finally {
            connection.disconnect()
        }
    }

    private fun open(path: String, method: String, authorized: Boolean): HttpURLConnection {
        val connection = URL("$endpoint$path").openConnection() as HttpURLConnection
        if (connection is HttpsURLConnection) configurePinnedTls(connection)
        connection.requestMethod = method
        connection.connectTimeout = 12_000
        connection.readTimeout = 5 * 60_000
        connection.useCaches = false
        connection.instanceFollowRedirects = false
        connection.setRequestProperty("user-agent", "OmniAGI-Android/1.0")
        if (authorized) {
            val currentToken = token ?: throw IllegalStateException("This phone is not paired.")
            connection.setRequestProperty("authorization", "Bearer $currentToken")
        }
        return connection
    }

    private fun readJsonResponse(connection: HttpURLConnection): JSONObject {
        val stream = if (connection.responseCode in 200..299) {
            connection.inputStream
        } else {
            connection.errorStream
        }
        val body = stream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }.orEmpty()
        if (connection.responseCode !in 200..299) {
            val message = runCatching { JSONObject(body).optString("error") }
                .getOrNull()
                ?.takeIf { it.isNotBlank() }
                ?: "Gateway request failed (${connection.responseCode})."
            throw GatewayException(connection.responseCode, message)
        }
        return if (body.isBlank()) JSONObject() else JSONObject(body)
    }

    private fun ensureSuccess(connection: HttpURLConnection) {
        if (connection.responseCode in 200..299) return
        val body = connection.errorStream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }
        val message = runCatching { JSONObject(body.orEmpty()).optString("error") }
            .getOrNull()
            ?.takeIf { it.isNotBlank() }
            ?: "Gateway request failed (${connection.responseCode})."
        throw GatewayException(connection.responseCode, message)
    }

    companion object {
        private data class EndpointIdentity(
            val endpoint: String,
            val pairingAddress: String,
            val certificateSha256: String?,
        )

        fun normalizeEndpoint(value: String): String {
            return parseEndpoint(value).pairingAddress
        }

        private fun parseEndpoint(value: String): EndpointIdentity {
            val normalized = value.trim().trimEnd('/')
            val url = runCatching { URL(normalized) }.getOrElse {
                throw IllegalArgumentException("Enter the full companion address shown by Studio.")
            }
            val protocol = url.protocol.lowercase()
            require(protocol == "http" || protocol == "https") {
                "Only http:// or https:// companion addresses are supported."
            }
            require(
                    url.host.isNotBlank() &&
                    url.userInfo == null &&
                    (url.path.orEmpty().isBlank() || url.path == "/") &&
                    url.query == null
            ) {
                "Enter only the companion address shown by Studio."
            }
            val fingerprint = url.ref
                ?.let { Regex("^sha256=([a-fA-F0-9]{64})$").matchEntire(it)?.groupValues?.get(1) }
                ?.lowercase()
            if (protocol == "http") {
                require(url.ref == null && isLoopbackOrEmulator(url.host)) {
                    "HTTP companion access is limited to loopback or the Android emulator. Use Studio's pinned HTTPS address for LAN."
                }
            } else {
                require(fingerprint != null) {
                    "The HTTPS companion address must include Studio's SHA-256 certificate pin."
                }
            }
            val bareHost = url.host.trim('[', ']')
            val host = if (bareHost.contains(':')) "[$bareHost]" else bareHost
            val port = if (url.port >= 0) ":${url.port}" else ""
            val endpoint = "$protocol://$host$port"
            val pairingAddress = if (fingerprint == null) {
                endpoint
            } else {
                "$endpoint/#sha256=$fingerprint"
            }
            return EndpointIdentity(endpoint, pairingAddress, fingerprint)
        }

        private fun isLoopbackOrEmulator(host: String): Boolean {
            val value = host.lowercase().trim('[', ']')
            if (value == "localhost" || value == "::1" || value == "10.0.2.2") return true
            val octets = value.split('.').mapNotNull(String::toIntOrNull)
            if (octets.size != 4 || octets.any { it !in 0..255 }) return false
            return octets[0] == 127
        }

        private fun certificateFingerprint(certificate: java.security.cert.Certificate): String =
            MessageDigest.getInstance("SHA-256")
                .digest(certificate.encoded)
                .joinToString("") { "%02x".format(it.toInt() and 0xff) }

        private fun pinnedTrustManager(expected: String): X509TrustManager =
            object : X509TrustManager {
                override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()

                override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {
                    throw CertificateException("Client certificates are not accepted.")
                }

                override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {
                    val leaf = chain?.firstOrNull()
                        ?: throw CertificateException("Studio returned no TLS certificate.")
                    leaf.checkValidity()
                    if (certificateFingerprint(leaf) != expected) {
                        throw CertificateException("Studio TLS certificate pin mismatch.")
                    }
                }
            }

        internal fun parseChatFrame(value: JSONObject): ChatFrame? =
            Parser.parseChatFrame(value)

        private fun encodePath(value: String): String = Parser.encodePath(value)
    }

    private fun configurePinnedTls(connection: HttpsURLConnection) {
        val expected = certificateSha256
            ?: throw IllegalArgumentException("Pinned TLS identity is required for HTTPS.")
        val context = SSLContext.getInstance("TLS")
        context.init(null, arrayOf<TrustManager>(pinnedTrustManager(expected)), SecureRandom())
        connection.sslSocketFactory = context.socketFactory
        connection.hostnameVerifier = javax.net.ssl.HostnameVerifier { _, session ->
            runCatching {
                certificateFingerprint(session.peerCertificates.first()) == expected
            }.getOrDefault(false)
        }
    }

    private object Parser {
        internal fun parseChatFrame(value: JSONObject): ChatFrame? {
            return when (value.optString("type")) {
                "stream" -> {
                    val event = value.optJSONObject("event") ?: return null
                    when (event.optString("type")) {
                        "chat-token" -> ChatFrame.Token(event.optString("delta"))
                        "chat-state" -> ChatFrame.State(
                            event.optString("state", "working"),
                            event.optString("error").takeIf { it.isNotBlank() },
                        )
                        "chat-action" -> {
                            val action = event.optJSONObject("actionEvent")
                                ?.optJSONObject("action")
                            val label = action?.optString("kind")
                                ?.takeIf { it.isNotBlank() }
                                ?: "action"
                            ChatFrame.Action(label)
                        }
                        "modality-preview" -> {
                            val preview = event.optJSONObject("preview")
                            ChatFrame.Preview(
                                preview?.optString("statusLabel")?.takeIf { it.isNotBlank() }
                                    ?: "Imagination forming",
                                preview?.takeIf { it.has("progress") }?.optDouble("progress"),
                            )
                        }
                        else -> null
                    }
                }
                "result" -> ChatFrame.Result(
                    value.optJSONObject("brainMessage")?.optString("content").orEmpty(),
                )
                "error" -> ChatFrame.Failure(
                    value.optString("error", "The neural turn failed."),
                    value.optBoolean("cancelled", false),
                )
                else -> null
            }
        }

        fun encodePath(value: String): String = URLEncoder.encode(
            value,
            StandardCharsets.UTF_8.name(),
        ).replace("+", "%20")
    }
}
