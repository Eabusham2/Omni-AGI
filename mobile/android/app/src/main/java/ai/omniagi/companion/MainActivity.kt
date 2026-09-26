package ai.omniagi.companion

import android.app.Activity
import android.content.Intent
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.io.IOException
import java.util.ArrayDeque
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : Activity() {
    private val executor: ExecutorService = Executors.newCachedThreadPool()
    private val main = Handler(Looper.getMainLooper())
    private val queuedMessages = ArrayDeque<String>()

    private lateinit var pairPanel: View
    private lateinit var chatPanel: View
    private lateinit var endpointInput: EditText
    private lateinit var codeInput: EditText
    private lateinit var pairButton: Button
    private lateinit var pairProgress: ProgressBar
    private lateinit var pairStatus: TextView
    private lateinit var identityLabel: TextView
    private lateinit var connectionBadge: TextView
    private lateinit var brainSpinner: Spinner
    private lateinit var disconnectButton: Button
    private lateinit var messageScroll: ScrollView
    private lateinit var messageList: LinearLayout
    private lateinit var liveStatus: TextView
    private lateinit var learnProgress: ProgressBar
    private lateinit var attachButton: Button
    private lateinit var messageInput: EditText
    private lateinit var sendButton: Button
    private lateinit var stopButton: Button

    @Volatile private var client: OmniGatewayClient? = null
    private var brains: List<BrainItem> = emptyList()
    private var selectedBrain: BrainItem? = null
    private var loadedBrainId: String? = null
    private var chatActive = false
    private var activeTurnId: String? = null
    private var cancelRequested = false
    private var currentBrainBubble: TextView? = null
    private var currentBrainText = StringBuilder()
    private var learningActive = false

    private val preferences by lazy {
        getSharedPreferences("omni-companion-session", MODE_PRIVATE)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bindViews()
        configureUi()
        restoreSession()
    }

    private fun bindViews() {
        pairPanel = findViewById(R.id.pairPanel)
        chatPanel = findViewById(R.id.chatPanel)
        endpointInput = findViewById(R.id.endpointInput)
        codeInput = findViewById(R.id.codeInput)
        pairButton = findViewById(R.id.pairButton)
        pairProgress = findViewById(R.id.pairProgress)
        pairStatus = findViewById(R.id.pairStatus)
        identityLabel = findViewById(R.id.identityLabel)
        connectionBadge = findViewById(R.id.connectionBadge)
        brainSpinner = findViewById(R.id.brainSpinner)
        disconnectButton = findViewById(R.id.disconnectButton)
        messageScroll = findViewById(R.id.messageScroll)
        messageList = findViewById(R.id.messageList)
        liveStatus = findViewById(R.id.liveStatus)
        learnProgress = findViewById(R.id.learnProgress)
        attachButton = findViewById(R.id.attachButton)
        messageInput = findViewById(R.id.messageInput)
        sendButton = findViewById(R.id.sendButton)
        stopButton = findViewById(R.id.stopButton)
    }

    private fun configureUi() {
        pairButton.setOnClickListener { pair() }
        disconnectButton.setOnClickListener { disconnectLocally() }
        sendButton.setOnClickListener { sendOrQueue() }
        stopButton.setOnClickListener { stopCurrentTurn() }
        attachButton.setOnClickListener { chooseExperience() }
        messageInput.setOnEditorActionListener { _, actionId, event ->
            val send = actionId == EditorInfo.IME_ACTION_SEND ||
                (event?.keyCode == android.view.KeyEvent.KEYCODE_ENTER && event.isCtrlPressed)
            if (send) sendOrQueue()
            send
        }
        brainSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: android.widget.AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long,
            ) {
                val brain = brains.getOrNull(position) ?: return
                selectedBrain = brain
                identityLabel.text = "Same persistent brain · ${brain.name}"
                if (loadedBrainId != brain.id) loadHistory(brain)
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }
    }

    private fun restoreSession() {
        val endpoint = preferences.getString("endpoint", null)
        val token = preferences.getString("token", null)
        if (endpoint.isNullOrBlank() || token.isNullOrBlank()) {
            showPairPanel()
            return
        }
        endpointInput.setText(endpoint)
        val restored = runCatching { OmniGatewayClient(endpoint, token) }.getOrNull()
        if (restored == null) {
            clearSession()
            showPairPanel("Saved companion address was invalid. Pair again.")
            return
        }
        client = restored
        showChatPanel()
        loadBrains(restored, true)
    }

    private fun pair() {
        val endpoint = endpointInput.text.toString()
        val code = codeInput.text.toString()
        pairButton.isEnabled = false
        pairProgress.visibility = View.VISIBLE
        pairStatus.text = "Contacting Studio…"
        executor.execute {
            try {
                val pairingClient = OmniGatewayClient(endpoint)
                val deviceName = "${Build.MANUFACTURER} ${Build.MODEL}".trim()
                val session = pairingClient.pair(code, deviceName)
                preferences.edit()
                    .putString("endpoint", session.endpoint)
                    .putString("token", session.token)
                    .putString("deviceId", session.deviceId)
                    .apply()
                val pairedClient = OmniGatewayClient(session.endpoint, session.token)
                client = pairedClient
                main.post {
                    codeInput.text.clear()
                    pairProgress.visibility = View.GONE
                    pairButton.isEnabled = true
                    showChatPanel()
                    loadBrains(pairedClient, false)
                }
            } catch (error: Exception) {
                main.post {
                    pairProgress.visibility = View.GONE
                    pairButton.isEnabled = true
                    pairStatus.text = friendlyError(error)
                }
            }
        }
    }

    private fun loadBrains(gateway: OmniGatewayClient, restored: Boolean) {
        setStatus(if (restored) "Reconnecting to the same brain…" else "Loading your brain instances…")
        executor.execute {
            try {
                val result = gateway.listBrains()
                main.post {
                    brains = result
                    if (result.isEmpty()) {
                        setStatus("No ready brain exists in Studio yet.")
                        return@post
                    }
                    val adapter = ArrayAdapter(
                        this,
                        android.R.layout.simple_spinner_item,
                        result,
                    )
                    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                    brainSpinner.adapter = adapter
                    val previous = preferences.getString("brainId", null)
                    val position = result.indexOfFirst { it.id == previous }.takeIf { it >= 0 } ?: 0
                    brainSpinner.setSelection(position)
                    setStatus("Connected · the desktop and phone share one neural state")
                }
            } catch (error: Exception) {
                main.post {
                    if (error is GatewayException && error.status == 401) {
                        clearSession()
                        client = null
                        showPairPanel("This device pairing was revoked. Pair again in Studio.")
                    } else {
                        setStatus(friendlyError(error))
                    }
                }
            }
        }
    }

    private fun loadHistory(brain: BrainItem) {
        val gateway = client ?: return
        preferences.edit().putString("brainId", brain.id).apply()
        loadedBrainId = brain.id
        messageList.removeAllViews()
        setStatus("Loading this brain's continuous conversation…")
        executor.execute {
            try {
                val messages = gateway.listMessages(brain.id)
                main.post {
                    if (selectedBrain?.id != brain.id) return@post
                    messageList.removeAllViews()
                    messages.forEach { appendMessage(it.role, it.content, false) }
                    setStatus("Ready · ${messages.size} persisted messages")
                    messageScroll.post { messageScroll.fullScroll(View.FOCUS_DOWN) }
                }
            } catch (error: Exception) {
                main.post { setStatus(friendlyError(error)) }
            }
        }
    }

    private fun sendOrQueue() {
        val text = messageInput.text.toString().trim()
        if (text.isBlank()) return
        messageInput.text.clear()
        if (chatActive) {
            queuedMessages.addLast(text)
            setStatus("Queued ${queuedMessages.size} message${if (queuedMessages.size == 1) "" else "s"} · current turn continues")
            return
        }
        startChat(text)
    }

    private fun startChat(text: String) {
        val gateway = client ?: return
        val brain = selectedBrain ?: run {
            setStatus("Choose a ready brain first.")
            return
        }
        if (brain.readiness != "ready") {
            setStatus("This brain is still completing its initial learning.")
            return
        }
        chatActive = true
        cancelRequested = false
        val turnId = UUID.randomUUID().toString()
        activeTurnId = turnId
        stopButton.isEnabled = true
        appendMessage("human", text, true)
        currentBrainText = StringBuilder()
        currentBrainBubble = appendMessage("brain", "", true)
        setStatus("Neural turn started…")
        executor.execute {
            var failure: Exception? = null
            try {
                gateway.streamChat(brain.id, text, turnId) { frame ->
                    main.post { consumeFrame(frame) }
                }
            } catch (error: Exception) {
                failure = error
            }
            main.post { finishChat(failure) }
        }
    }

    private fun consumeFrame(frame: ChatFrame) {
        when (frame) {
            is ChatFrame.Token -> {
                val follow = isNearBottom()
                currentBrainText.append(frame.delta)
                currentBrainBubble?.text = currentBrainText.toString()
                if (follow) messageScroll.post { messageScroll.fullScroll(View.FOCUS_DOWN) }
            }
            is ChatFrame.State -> setStatus(
                frame.detail ?: when (frame.state) {
                    "started" -> "Neural activity settling…"
                    "complete" -> "Turn committed to the same brain"
                    "cancelled" -> "Turn stopped"
                    else -> frame.state.replaceFirstChar { it.uppercase() }
                },
            )
            is ChatFrame.Action -> setStatus("Brain action · ${frame.label}")
            is ChatFrame.Preview -> {
                val percent = frame.progress?.let { " · ${(it * 100).roundToInt()}%" }.orEmpty()
                setStatus("${frame.label}$percent")
            }
            is ChatFrame.Result -> {
                if (frame.content.isNotBlank()) {
                    currentBrainText = StringBuilder(frame.content)
                    currentBrainBubble?.text = frame.content
                }
            }
            is ChatFrame.Failure -> {
                setStatus(frame.message)
                if (currentBrainText.isEmpty()) currentBrainBubble?.text = if (frame.cancelled) "Stopped." else frame.message
            }
        }
    }

    private fun finishChat(error: Exception?) {
        val wasCancelled = cancelRequested
        chatActive = false
        activeTurnId = null
        stopButton.isEnabled = false
        currentBrainBubble = null
        if (error != null && !wasCancelled) {
            setStatus(friendlyError(error))
            if (currentBrainText.isEmpty()) appendActivityCard("The turn did not complete: ${friendlyError(error)}")
        } else if (wasCancelled) {
            setStatus("Turn stopped · retained committed neural changes only")
        } else {
            setStatus("Ready${if (queuedMessages.isEmpty()) "" else " · opening queued message"}")
        }
        cancelRequested = false
        val next = queuedMessages.pollFirst()
        if (next != null) startChat(next)
    }

    private fun stopCurrentTurn() {
        val gateway = client ?: return
        val brainId = selectedBrain?.id ?: return
        val turnId = activeTurnId ?: return
        cancelRequested = true
        queuedMessages.clear()
        setStatus("Stopping this turn…")
        gateway.disconnectActiveChat()
        executor.execute {
            runCatching { gateway.cancel(brainId, turnId) }
        }
    }

    private fun chooseExperience() {
        if (learningActive) return
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(
                Intent.EXTRA_MIME_TYPES,
                arrayOf(
                    "application/pdf",
                    "text/*",
                    "image/*",
                    "audio/*",
                    "video/*",
                    "application/json",
                    "application/zip",
                    "application/octet-stream",
                ),
            )
        }
        startActivityForResult(intent, EXPERIENCE_REQUEST)
    }

    @Deprecated("Kept for API 26 compatibility with the framework picker")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != EXPERIENCE_REQUEST || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        runCatching {
            contentResolver.takePersistableUriPermission(
                uri,
                data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        }
        learnExperience(uri)
    }

    private fun learnExperience(uri: Uri) {
        val gateway = client ?: return
        val brain = selectedBrain ?: return
        val metadata = fileMetadata(uri)
        learningActive = true
        attachButton.isEnabled = false
        learnProgress.visibility = View.VISIBLE
        learnProgress.progress = 0
        setStatus("Reading ${metadata.name} without loading it all into RAM…")
        executor.execute {
            try {
                val input = contentResolver.openInputStream(uri)
                    ?: throw IOException("Android could not open this experience.")
                val result = gateway.uploadExperience(
                    brainId = brain.id,
                    fileName = metadata.name,
                    mimeType = metadata.mimeType,
                    declaredBytes = metadata.bytes,
                    input = input,
                ) { written, total ->
                    if (total != null && total > 0) {
                        val progress = ((written.toDouble() / total.toDouble()) * 100).roundToInt()
                            .coerceIn(0, 100)
                        main.post {
                            learnProgress.progress = progress
                            setStatus("Streaming ${metadata.name} · $progress%")
                        }
                    }
                }
                main.post {
                    val learned = result.optJSONArray("results")?.length() ?: 0
                    appendActivityCard("Learned ${metadata.name} through the full neural ingestion path · $learned source receipt${if (learned == 1) "" else "s"}")
                    setStatus("Experience committed to parameters and synapses")
                    finishLearning()
                }
            } catch (error: Exception) {
                main.post {
                    appendActivityCard("Learning paused: ${friendlyError(error)}")
                    setStatus(friendlyError(error))
                    finishLearning()
                }
            }
        }
    }

    private fun finishLearning() {
        learningActive = false
        attachButton.isEnabled = true
        learnProgress.visibility = View.GONE
    }

    private data class FileMetadata(val name: String, val bytes: Long?, val mimeType: String)

    private fun fileMetadata(uri: Uri): FileMetadata {
        var name = "mobile-experience"
        var bytes: Long? = null
        contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    .takeIf { it >= 0 }
                    ?.let { name = cursor.getString(it) ?: name }
                cursor.getColumnIndex(OpenableColumns.SIZE)
                    .takeIf { it >= 0 && !cursor.isNull(it) }
                    ?.let { bytes = cursor.getLong(it) }
            }
        }
        return FileMetadata(
            name = name,
            bytes = bytes,
            mimeType = contentResolver.getType(uri) ?: "application/octet-stream",
        )
    }

    private fun appendMessage(role: String, content: String, scroll: Boolean): TextView {
        val wrapper = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = if (role == "human") Gravity.END else Gravity.START
            setPadding(0, dp(4), 0, dp(4))
        }
        val bubble = TextView(this).apply {
            text = content
            setTextColor(
                ContextCompat.getColor(
                    this@MainActivity,
                    if (role == "human") R.color.omni_on_accent else R.color.omni_text,
                ),
            )
            textSize = 14f
            setLineSpacing(0f, 1.12f)
            setPadding(dp(13), dp(10), dp(13), dp(10))
            maxWidth = (resources.displayMetrics.widthPixels * 0.86).roundToInt()
            background = GradientDrawable().apply {
                cornerRadius = dp(16).toFloat()
                setColor(
                    ContextCompat.getColor(
                        this@MainActivity,
                        if (role == "human") R.color.omni_human_bubble else R.color.omni_brain_bubble,
                    ),
                )
                if (role != "human") {
                    setStroke(dp(1), ContextCompat.getColor(this@MainActivity, R.color.omni_border))
                }
            }
            if (content.isBlank()) contentDescription = "Brain response streaming"
        }
        wrapper.addView(
            bubble,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        messageList.addView(wrapper)
        if (scroll) messageScroll.post { messageScroll.fullScroll(View.FOCUS_DOWN) }
        return bubble
    }

    private fun appendActivityCard(content: String) {
        val card = TextView(this).apply {
            text = content
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.omni_muted))
            textSize = 10.5f
            setTypeface(typeface, Typeface.BOLD)
            setPadding(dp(11), dp(8), dp(11), dp(8))
            background = GradientDrawable().apply {
                cornerRadius = dp(10).toFloat()
                setColor(ContextCompat.getColor(this@MainActivity, R.color.omni_accent_soft))
            }
        }
        val wrapper = LinearLayout(this).apply {
            gravity = Gravity.START
            setPadding(0, dp(4), 0, dp(4))
            addView(card)
        }
        messageList.addView(wrapper)
        messageScroll.post { messageScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun isNearBottom(): Boolean {
        val child = messageScroll.getChildAt(0) ?: return true
        return child.bottom - (messageScroll.height + messageScroll.scrollY) < dp(72)
    }

    private fun showPairPanel(message: String = "") {
        pairPanel.visibility = View.VISIBLE
        chatPanel.visibility = View.GONE
        connectionBadge.text = "OFFLINE"
        identityLabel.text = "Same persistent brain · companion disconnected"
        pairStatus.text = message
    }

    private fun showChatPanel() {
        pairPanel.visibility = View.GONE
        chatPanel.visibility = View.VISIBLE
        connectionBadge.text = "PAIRED"
    }

    private fun disconnectLocally() {
        client?.disconnectActiveChat()
        client = null
        clearSession()
        brains = emptyList()
        selectedBrain = null
        loadedBrainId = null
        queuedMessages.clear()
        chatActive = false
        showPairPanel("Local pairing removed. Revoke this device in Studio if it should never reconnect.")
    }

    private fun clearSession() {
        preferences.edit().clear().apply()
    }

    private fun setStatus(value: String) {
        liveStatus.text = value
    }

    private fun friendlyError(error: Exception): String = when (error) {
        is GatewayException -> error.message ?: "Studio rejected the request."
        is IllegalArgumentException -> error.message ?: "Check the companion address and code."
        else -> error.message?.takeIf { it.isNotBlank() }
            ?: "Could not reach Studio. Keep it open and check the companion address."
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).roundToInt()

    override fun onDestroy() {
        client?.disconnectActiveChat()
        executor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val EXPERIENCE_REQUEST = 7101
    }
}
