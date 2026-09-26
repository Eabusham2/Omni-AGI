import XCTest
@testable import OmniCompanion

final class GatewayClientTests: XCTestCase {
    private actor FrameCollector {
        var values: [ChatFrame] = []
        func append(_ value: ChatFrame) { values.append(value) }
    }

    private final class ProgressCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Int64 = 0
        func set(_ value: Int64) { lock.lock(); stored = value; lock.unlock() }
        func get() -> Int64 { lock.lock(); defer { lock.unlock() }; return stored }
    }

    func testEndpointAllowsOnlyLoopbackHTTPAndRequiresPinnedLANHTTPS() throws {
        let pin = String(repeating: "ab", count: 32)
        XCTAssertEqual(try GatewayClient.normalizedEndpoint("http://127.0.0.1:41837/").absoluteString, "http://127.0.0.1:41837")
        XCTAssertEqual(
            try GatewayClient.normalizedEndpoint("https://192.168.1.20:41837/#sha256=\(pin.uppercased())").absoluteString,
            "https://192.168.1.20:41837/#sha256=\(pin)"
        )
        XCTAssertThrowsError(try GatewayClient.normalizedEndpoint("http://example.com:41837"))
        XCTAssertThrowsError(try GatewayClient.normalizedEndpoint("http://192.168.1.20:41837"))
        XCTAssertThrowsError(try GatewayClient.normalizedEndpoint("https://192.168.1.20:41837"))
        XCTAssertThrowsError(try GatewayClient.normalizedEndpoint("http://user@127.0.0.1:41837"))
        XCTAssertThrowsError(try GatewayClient.normalizedEndpoint("http://127.0.0.1:41837/hidden"))
    }

    func testParsesStreamFramesWithoutBehavioralPromptText() throws {
        let token = Data(#"{"type":"stream","event":{"type":"chat-token","delta":"hello"}}"#.utf8)
        let action = Data(#"{"type":"stream","event":{"type":"chat-action","actionEvent":{"action":{"kind":"imagine"}}}}"#.utf8)
        let result = Data(#"{"type":"result","brainMessage":{"content":"whole reply"}}"#.utf8)
        XCTAssertEqual(GatewayClient.parseChatFrame(token), .token("hello"))
        XCTAssertEqual(GatewayClient.parseChatFrame(action), .action("imagine"))
        XCTAssertEqual(GatewayClient.parseChatFrame(result), .result("whole reply"))
    }

    func testHostGatewayRoundTripWhenProvided() async throws {
        guard let endpoint = ProcessInfo.processInfo.environment["OMNI_GATEWAY_ENDPOINT"] else {
            throw XCTSkip("The host fake gateway is only present in simulator CI.")
        }
        let paired = try await GatewayClient(endpoint: endpoint).pair(code: "123456", deviceName: "iOS unit test")
        let client = try GatewayClient(endpoint: paired.endpoint.absoluteString, token: paired.token)
        let listedBrains = try await client.listBrains()
        let brain = try XCTUnwrap(listedBrains.first)
        XCTAssertEqual(brain.id, "brain-1")
        let history = try await client.listMessages(brainID: brain.id)
        XCTAssertEqual(history.first?.content, "Existing persisted conversation")

        let frames = FrameCollector()
        try await client.streamChat(brainID: brain.id, input: "Hello", turnID: UUID().uuidString) { frame in
            await frames.append(frame)
        }
        let receivedFrames = await frames.values
        XCTAssertTrue(receivedFrames.contains(.token("Same-brain emulator reply")))
        XCTAssertTrue(receivedFrames.contains(.result("Same-brain emulator reply")))

        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("omni-ios-upload-\(UUID().uuidString).raw")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try Data(repeating: 0x5a, count: 2 * 1024 * 1024).write(to: temporary, options: .atomic)
        let progress = ProgressCollector()
        let receipt = try await client.uploadExperience(
            brainID: brain.id,
            fileURL: temporary,
            fileName: "simulator-audio.raw",
            mimeType: "audio/x-raw"
        ) { sent, _ in progress.set(sent) }
        XCTAssertEqual(receipt.bytes, 2 * 1024 * 1024)
        XCTAssertEqual(receipt.sourceCount, 1)
        XCTAssertEqual(progress.get(), 2 * 1024 * 1024)
    }
}
