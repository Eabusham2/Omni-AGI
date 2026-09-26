import Foundation

private actor FrameCollector {
    var frames: [ChatFrame] = []
    func append(_ frame: ChatFrame) { frames.append(frame) }
}

private final class ProgressCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Int64 = 0
    func update(_ next: Int64) { lock.lock(); value = next; lock.unlock() }
    func read() -> Int64 { lock.lock(); defer { lock.unlock() }; return value }
}

@main
struct CompanionProtocolSmoke {
    static func main() async throws {
        let endpoint = CommandLine.arguments.dropFirst().first ?? "http://127.0.0.1:41837"
        let paired = try await GatewayClient(endpoint: endpoint).pair(code: "123456", deviceName: "Swift protocol smoke")
        let client = try GatewayClient(endpoint: paired.endpoint.absoluteString, token: paired.token)
        let brains = try await client.listBrains()
        guard let brain = brains.first, brain.id == "brain-1" else {
            throw GatewayError(status: nil, message: "The same-brain listing was not returned.")
        }
        let history = try await client.listMessages(brainID: brain.id)
        guard history.first?.content == "Existing persisted conversation" else {
            throw GatewayError(status: nil, message: "Continuous history was not returned.")
        }

        let frames = FrameCollector()
        try await client.streamChat(brainID: brain.id, input: "Hello from Swift", turnID: UUID().uuidString) { frame in
            await frames.append(frame)
        }
        let received = await frames.frames
        guard received.contains(.token("Same-brain emulator reply")),
              received.contains(.result("Same-brain emulator reply")) else {
            throw GatewayError(status: nil, message: "The streamed neural reply was incomplete.")
        }

        let upload = FileManager.default.temporaryDirectory.appendingPathComponent("omni-ios-smoke-\(UUID().uuidString).raw")
        defer { try? FileManager.default.removeItem(at: upload) }
        try Data(repeating: 0x5a, count: 2 * 1024 * 1024).write(to: upload, options: .atomic)
        let progress = ProgressCollector()
        let receipt = try await client.uploadExperience(
            brainID: brain.id,
            fileURL: upload,
            fileName: "swift-audio.raw",
            mimeType: "audio/x-raw"
        ) { sent, _ in progress.update(sent) }
        guard receipt.bytes == 2 * 1024 * 1024,
              receipt.sourceCount == 1,
              progress.read() == 2 * 1024 * 1024 else {
            throw GatewayError(status: nil, message: "The streamed experience upload was incomplete.")
        }
        print("iOS companion protocol smoke passed: pair, list, history, NDJSON chat, and 2 MiB file learning.")
    }
}
