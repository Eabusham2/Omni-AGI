import Foundation
import Combine
import UIKit
import UniformTypeIdentifiers

@MainActor
final class CompanionViewModel: ObservableObject {
    @Published var endpointText = "http://127.0.0.1:41837"
    @Published var pairingCode = ""
    @Published private(set) var isPaired = false
    @Published private(set) var isConnecting = false
    @Published private(set) var brains: [BrainSummary] = []
    @Published var selectedBrainID = "" {
        didSet {
            guard selectedBrainID != oldValue, !selectedBrainID.isEmpty else { return }
            defaults.set(selectedBrainID, forKey: Keys.brainID)
            Task { await loadHistory() }
        }
    }
    @Published private(set) var messages: [DisplayMessage] = []
    @Published var draft = ""
    @Published private(set) var status = "Pair this device to the same persistent brain in Studio."
    @Published private(set) var isChatting = false
    @Published private(set) var queuedCount = 0
    @Published private(set) var isLearning = false
    @Published private(set) var learningProgress: Double?

    private enum Keys {
        static let endpoint = "omni.companion.endpoint"
        static let brainID = "omni.companion.brain-id"
        static let deviceID = "omni.companion.device-id"
        static let token = "mobile-gateway-token"
    }

    private let defaults: UserDefaults
    private var client: GatewayClient?
    private var queuedInputs: [String] = []
    private var activeTurnID: String?
    private var activeBrainMessageID: String?
    private var activeChatTask: Task<Void, Never>?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("--reset-pairing") {
            KeychainStore.delete(account: Keys.token)
            defaults.removeObject(forKey: Keys.endpoint)
            defaults.removeObject(forKey: Keys.brainID)
            defaults.removeObject(forKey: Keys.deviceID)
        }
        endpointText = Self.argument(after: "--gateway-endpoint", in: arguments)
            ?? defaults.string(forKey: Keys.endpoint)
            ?? endpointText
        pairingCode = Self.argument(after: "--pair-code", in: arguments) ?? ""
        Task { await restoreSession() }
    }

    var selectedBrain: BrainSummary? {
        brains.first(where: { $0.id == selectedBrainID })
    }

    var selectedBrainName: String {
        selectedBrain?.name ?? "Persistent brain"
    }

    func pair() {
        guard !isConnecting else { return }
        isConnecting = true
        status = "Contacting Studio…"
        Task {
            defer { isConnecting = false }
            do {
                let pairingClient = try GatewayClient(endpoint: endpointText)
                let session = try await pairingClient.pair(code: pairingCode, deviceName: UIDevice.current.name)
                try KeychainStore.save(session.token, account: Keys.token)
                defaults.set(session.endpoint.absoluteString, forKey: Keys.endpoint)
                defaults.set(session.deviceID, forKey: Keys.deviceID)
                endpointText = session.endpoint.absoluteString
                pairingCode = ""
                client = try GatewayClient(endpoint: endpointText, token: session.token)
                isPaired = true
                try await loadBrains()
            } catch {
                status = Self.friendly(error)
            }
        }
    }

    func disconnect() {
        activeChatTask?.cancel()
        activeChatTask = nil
        KeychainStore.delete(account: Keys.token)
        defaults.removeObject(forKey: Keys.brainID)
        defaults.removeObject(forKey: Keys.deviceID)
        client = nil
        brains = []
        messages = []
        selectedBrainID = ""
        queuedInputs = []
        queuedCount = 0
        isChatting = false
        isPaired = false
        status = "Disconnected locally. Revoke the device in Studio to invalidate it everywhere."
    }

    func refreshBrains() {
        Task {
            do { try await loadBrains() }
            catch { handle(error) }
        }
    }

    func sendOrQueue() {
        let input = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return }
        draft = ""
        if isChatting {
            queuedInputs.append(input)
            queuedCount = queuedInputs.count
            status = "Queued \(queuedCount) message\(queuedCount == 1 ? "" : "s"); the current turn continues."
            return
        }
        startChat(input)
    }

    func stopTurn() {
        guard let client, let brainID = selectedBrain?.id, let turnID = activeTurnID else { return }
        queuedInputs.removeAll()
        queuedCount = 0
        status = "Stopping this turn…"
        activeChatTask?.cancel()
        Task { _ = try? await client.cancel(brainID: brainID, turnID: turnID) }
    }

    func learn(files: [URL]) {
        guard !isLearning, let client, let brain = selectedBrain else { return }
        isLearning = true
        learningProgress = 0
        status = "Opening \(files.count) experience\(files.count == 1 ? "" : "s")…"
        Task {
            defer {
                isLearning = false
                learningProgress = nil
            }
            var completed = 0
            for fileURL in files {
                let scoped = fileURL.startAccessingSecurityScopedResource()
                defer { if scoped { fileURL.stopAccessingSecurityScopedResource() } }
                do {
                    let values = try fileURL.resourceValues(forKeys: [.nameKey, .fileSizeKey, .contentTypeKey])
                    let name = values.name ?? fileURL.lastPathComponent
                    let total = Int64(values.fileSize ?? 0)
                    status = "Learning from \(name)…"
                    _ = try await client.uploadExperience(
                        brainID: brain.id,
                        fileURL: fileURL,
                        fileName: name,
                        mimeType: values.contentType?.preferredMIMEType ?? "application/octet-stream"
                    ) { [weak self] sent, expected in
                        Task { @MainActor in
                            guard let self else { return }
                            let denominator = expected > 0 ? expected : total
                            let withinFile = denominator > 0 ? min(1, Double(sent) / Double(denominator)) : 0
                            self.learningProgress = (Double(completed) + withinFile) / Double(max(files.count, 1))
                        }
                    }
                    completed += 1
                    learningProgress = Double(completed) / Double(max(files.count, 1))
                } catch {
                    status = "Learning paused at \(fileURL.lastPathComponent): \(Self.friendly(error))"
                    return
                }
            }
            status = "Learned from all \(completed) selected experience\(completed == 1 ? "" : "s") in the same brain."
        }
    }

    private func restoreSession() async {
        guard let endpoint = defaults.string(forKey: Keys.endpoint),
              let token = KeychainStore.read(account: Keys.token) else { return }
        do {
            client = try GatewayClient(endpoint: endpoint, token: token)
            endpointText = endpoint
            isPaired = true
            status = "Reconnecting to the same brain…"
            try await loadBrains()
        } catch {
            handle(error)
        }
    }

    private func loadBrains() async throws {
        guard let client else { return }
        let available = try await client.listBrains()
        brains = available
        guard !available.isEmpty else {
            selectedBrainID = ""
            messages = []
            status = "No brain exists in Studio yet. Create and train one on the desktop."
            return
        }
        let previous = defaults.string(forKey: Keys.brainID)
        selectedBrainID = available.contains(where: { $0.id == previous }) ? previous! : available[0].id
        if selectedBrainID == previous { await loadHistory() }
        status = "Connected; desktop and mobile share one neural state."
    }

    private func loadHistory() async {
        guard let client, !selectedBrainID.isEmpty else { return }
        let requestedID = selectedBrainID
        status = "Loading this brain’s continuous conversation…"
        do {
            let history = try await client.listMessages(brainID: requestedID)
            guard requestedID == selectedBrainID else { return }
            messages = history.map { DisplayMessage(id: $0.id, role: $0.role, content: $0.content, pending: false) }
            status = "Ready; \(history.count) persisted message\(history.count == 1 ? "" : "s")."
        } catch {
            handle(error)
        }
    }

    private func startChat(_ input: String) {
        guard let client, let brain = selectedBrain else {
            status = "Choose a ready brain first."
            return
        }
        guard brain.readiness == "ready" else {
            status = "This brain is still completing its initial learning."
            return
        }
        let turnID = UUID().uuidString
        let replyID = "pending-\(turnID)"
        activeTurnID = turnID
        activeBrainMessageID = replyID
        isChatting = true
        messages.append(DisplayMessage(id: "local-human-\(turnID)", role: "human", content: input, pending: false))
        messages.append(DisplayMessage(id: replyID, role: "brain", content: "", pending: true))
        status = "Neural turn started…"
        activeChatTask = Task { [weak self] in
            guard let self else { return }
            var caught: Error?
            do {
                try await client.streamChat(brainID: brain.id, input: input, turnID: turnID) { [weak self] frame in
                    await self?.consume(frame, replyID: replyID)
                }
            } catch {
                caught = error
            }
            self.finishTurn(error: caught, replyID: replyID)
        }
    }

    private func consume(_ frame: ChatFrame, replyID: String) {
        guard let index = messages.firstIndex(where: { $0.id == replyID }) else { return }
        switch frame {
        case .token(let delta):
            messages[index].content += delta
        case .state(let state, let detail):
            status = detail ?? Self.stateLabel(state)
        case .action(let label):
            status = "Brain action: \(label)"
        case .preview(let label, let progress):
            let suffix = progress.map { " · \(Int($0 * 100))%" } ?? ""
            status = label + suffix
        case .result(let content):
            if !content.isEmpty { messages[index].content = content }
        case .failure(let message, let cancelled):
            status = message
            if messages[index].content.isEmpty { messages[index].content = cancelled ? "Stopped." : message }
        }
    }

    private func finishTurn(error: Error?, replyID: String) {
        if let index = messages.firstIndex(where: { $0.id == replyID }) {
            messages[index].pending = false
            if messages[index].content.isEmpty {
                messages[index].content = error is CancellationError ? "Stopped." : (error.map(Self.friendly) ?? "No reply was produced.")
            }
        }
        isChatting = false
        activeTurnID = nil
        activeBrainMessageID = nil
        activeChatTask = nil
        if let error, !(error is CancellationError) { handle(error) }
        else if error is CancellationError { status = "Turn stopped; committed neural changes were retained." }
        else { status = "Ready. The completed turn remains part of this brain’s learning stream." }
        if !queuedInputs.isEmpty {
            let next = queuedInputs.removeFirst()
            queuedCount = queuedInputs.count
            startChat(next)
        }
    }

    private func handle(_ error: Error) {
        status = Self.friendly(error)
        if (error as? GatewayError)?.status == 401 {
            KeychainStore.delete(account: Keys.token)
            client = nil
            isPaired = false
            brains = []
            messages = []
            status = "This device pairing was revoked. Pair it again in Studio."
        }
    }

    private static func friendly(_ error: Error) -> String {
        if let gateway = error as? GatewayError { return gateway.message }
        if error is CancellationError { return "Stopped."
        }
        let network = error as NSError
        if network.domain == NSURLErrorDomain {
            return "Could not reach Studio. Confirm the desktop gateway is running and both devices use the same network."
        }
        return error.localizedDescription
    }

    private static func stateLabel(_ state: String) -> String {
        switch state {
        case "started": return "Neural activity settling…"
        case "complete": return "Turn committed to the same brain."
        case "cancelled": return "Turn stopped."
        default: return state.replacingOccurrences(of: "-", with: " ").capitalized
        }
    }

    private static func argument(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }
}
