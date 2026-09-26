import Foundation
import CryptoKit
import Security

final class GatewayClient: @unchecked Sendable {
    let endpoint: URL
    private let pairingEndpoint: URL
    private let token: String?
    private let session: URLSession
    private let pinningDelegate: CertificatePinningDelegate?
    private let certificateSha256: String?

    init(endpoint: String, token: String? = nil) throws {
        let identity = try Self.endpointIdentity(endpoint)
        self.endpoint = identity.endpoint
        self.pairingEndpoint = identity.pairingEndpoint
        self.token = token
        self.certificateSha256 = identity.certificateSha256
        if let fingerprint = identity.certificateSha256 {
            let delegate = CertificatePinningDelegate(fingerprint: fingerprint)
            let configuration = URLSessionConfiguration.ephemeral
            configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
            self.pinningDelegate = delegate
            self.session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        } else {
            self.pinningDelegate = nil
            self.session = .shared
        }
    }

    static func normalizedEndpoint(_ value: String) throws -> URL {
        try endpointIdentity(value).pairingEndpoint
    }

    private struct EndpointIdentity {
        let endpoint: URL
        let pairingEndpoint: URL
        let certificateSha256: String?
    }

    private static func endpointIdentity(_ value: String) throws -> EndpointIdentity {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var parts = URLComponents(string: trimmed),
              let scheme = parts.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              parts.user == nil,
              parts.password == nil,
              let host = parts.host,
              !host.isEmpty,
              parts.query == nil,
              parts.path.isEmpty || parts.path == "/" else {
            throw GatewayError(status: nil, message: "Enter only the full companion address shown by Studio.")
        }
        let fingerprint: String? = {
            guard let fragment = parts.fragment,
                  let match = fragment.range(of: #"^sha256=[a-fA-F0-9]{64}$"#, options: .regularExpression) else {
                return nil
            }
            return String(fragment[match]).dropFirst("sha256=".count).lowercased()
        }()
        if scheme == "http" && (parts.fragment != nil || !isLoopback(host)) {
            throw GatewayError(status: nil, message: "HTTP companion access is limited to loopback. Use Studio's pinned HTTPS address for LAN.")
        }
        if scheme == "https" && fingerprint == nil {
            throw GatewayError(status: nil, message: "The HTTPS companion address must include Studio's SHA-256 certificate pin.")
        }
        parts.path = ""
        parts.fragment = nil
        guard let endpoint = parts.url else {
            throw GatewayError(status: nil, message: "The companion address is invalid.")
        }
        var pairingParts = parts
        pairingParts.path = fingerprint == nil ? "" : "/"
        pairingParts.fragment = fingerprint.map { "sha256=\($0)" }
        guard let pairingEndpoint = pairingParts.url else {
            throw GatewayError(status: nil, message: "The companion pairing identity is invalid.")
        }
        return EndpointIdentity(
            endpoint: endpoint,
            pairingEndpoint: pairingEndpoint,
            certificateSha256: fingerprint
        )
    }

    private static func isLoopback(_ host: String) -> Bool {
        let value = host.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        if value == "localhost" || value == "::1" { return true }
        let octets = value.split(separator: ".").compactMap { Int($0) }
        guard octets.count == 4, octets.allSatisfy({ 0...255 ~= $0 }) else { return false }
        return octets[0] == 127
    }

    func pair(code: String, deviceName: String) async throws -> PairedSession {
        guard code.range(of: #"^\d{6}$"#, options: .regularExpression) != nil else {
            throw GatewayError(status: nil, message: "Enter the six-digit pairing code.")
        }
        let value = try await requestJSON(
            path: "/v1/pair",
            method: "POST",
            body: ["code": code, "deviceName": deviceName],
            authorized: false
        )
        guard let token = value["token"] as? String,
              let device = value["device"] as? [String: Any],
              let deviceID = device["id"] as? String else {
            throw GatewayError(status: nil, message: "Studio returned an incomplete pairing response.")
        }
        let responseFingerprint = (value["certificateSha256"] as? String)?.lowercased()
        if let certificateSha256, responseFingerprint != certificateSha256 {
            throw GatewayError(status: 495, message: "Studio's pairing identity did not match the pinned TLS certificate.")
        }
        return PairedSession(
            endpoint: pairingEndpoint,
            token: token,
            deviceID: deviceID,
            certificateSha256: responseFingerprint
        )
    }

    func listBrains() async throws -> [BrainSummary] {
        let value = try await requestJSON(path: "/v1/brains", method: "GET", authorized: true)
        guard let raw = value["brains"] else { return [] }
        let data = try JSONSerialization.data(withJSONObject: raw)
        return try JSONDecoder().decode([BrainSummary].self, from: data)
    }

    func listMessages(brainID: String) async throws -> [GatewayMessage] {
        let value = try await requestJSON(
            path: "/v1/brains/\(Self.path(brainID))/messages",
            method: "GET",
            authorized: true
        )
        guard let raw = value["messages"] else { return [] }
        let data = try JSONSerialization.data(withJSONObject: raw)
        return try JSONDecoder().decode([GatewayMessage].self, from: data)
    }

    func streamChat(
        brainID: String,
        input: String,
        turnID: String,
        onFrame: @escaping @Sendable (ChatFrame) async -> Void
    ) async throws {
        var request = try makeRequest(
            path: "/v1/brains/\(Self.path(brainID))/chat",
            method: "POST",
            authorized: true
        )
        request.setValue("application/json; charset=utf-8", forHTTPHeaderField: "content-type")
        request.setValue("application/x-ndjson", forHTTPHeaderField: "accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["input": input, "turnId": turnID])
        let (bytes, response) = try await session.bytes(for: request)
        try Self.ensureSuccess(response: response, data: nil)
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard !line.isEmpty, let data = line.data(using: .utf8),
                  let frame = Self.parseChatFrame(data) else { continue }
            await onFrame(frame)
        }
    }

    func cancel(brainID: String, turnID: String) async throws -> Int {
        let value = try await requestJSON(
            path: "/v1/brains/\(Self.path(brainID))/chat/\(Self.path(turnID))/cancel",
            method: "POST",
            body: [:],
            authorized: true
        )
        return value["cancelled"] as? Int ?? 0
    }

    func uploadExperience(
        brainID: String,
        fileURL: URL,
        fileName: String,
        mimeType: String,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> UploadReceipt {
        var request = try makeRequest(
            path: "/v1/brains/\(Self.path(brainID))/experience",
            method: "POST",
            authorized: true
        )
        request.setValue(mimeType.isEmpty ? "application/octet-stream" : mimeType, forHTTPHeaderField: "content-type")
        request.setValue(fileName.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? fileName, forHTTPHeaderField: "x-omni-filename")
        request.setValue("application/json", forHTTPHeaderField: "accept")
        let bridge = UploadBridge(
            certificateSha256: certificateSha256,
            onProgress: onProgress
        )
        let (data, response) = try await bridge.upload(request: request, fileURL: fileURL)
        try Self.ensureSuccess(response: response, data: data)
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GatewayError(status: nil, message: "Studio returned an invalid learning receipt.")
        }
        return UploadReceipt(
            fileName: value["fileName"] as? String ?? fileName,
            bytes: (value["bytes"] as? NSNumber)?.int64Value ?? 0,
            sourceCount: (value["results"] as? [Any])?.count ?? 0
        )
    }

    private func requestJSON(
        path: String,
        method: String,
        body: [String: Any]? = nil,
        authorized: Bool
    ) async throws -> [String: Any] {
        var request = try makeRequest(path: path, method: method, authorized: authorized)
        request.setValue("application/json", forHTTPHeaderField: "accept")
        if let body {
            request.setValue("application/json; charset=utf-8", forHTTPHeaderField: "content-type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await session.data(for: request)
        try Self.ensureSuccess(response: response, data: data)
        if data.isEmpty { return [:] }
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GatewayError(status: nil, message: "Studio returned an invalid response.")
        }
        return value
    }

    private func makeRequest(path: String, method: String, authorized: Bool) throws -> URLRequest {
        guard let url = URL(string: path, relativeTo: endpoint)?.absoluteURL else {
            throw GatewayError(status: nil, message: "The companion request address is invalid.")
        }
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 300)
        request.httpMethod = method
        request.setValue("OmniAGI-iOS/1.0", forHTTPHeaderField: "user-agent")
        if authorized {
            guard let token, !token.isEmpty else {
                throw GatewayError(status: nil, message: "This device is not paired.")
            }
            request.setValue("Bearer \(token)", forHTTPHeaderField: "authorization")
        }
        return request
    }

    static func parseChatFrame(_ data: Data) -> ChatFrame? {
        guard let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = value["type"] as? String else { return nil }
        if type == "result" {
            let message = value["brainMessage"] as? [String: Any]
            return .result(message?["content"] as? String ?? "")
        }
        if type == "error" {
            return .failure(value["error"] as? String ?? "The neural turn failed.", value["cancelled"] as? Bool ?? false)
        }
        guard type == "stream", let event = value["event"] as? [String: Any],
              let eventType = event["type"] as? String else { return nil }
        switch eventType {
        case "chat-token": return .token(event["delta"] as? String ?? "")
        case "chat-state": return .state(event["state"] as? String ?? "working", event["error"] as? String)
        case "chat-action":
            let actionEvent = event["actionEvent"] as? [String: Any]
            let action = actionEvent?["action"] as? [String: Any]
            return .action(action?["kind"] as? String ?? "action")
        case "modality-preview":
            let preview = event["preview"] as? [String: Any]
            return .preview(preview?["statusLabel"] as? String ?? "Imagination forming", (preview?["progress"] as? NSNumber)?.doubleValue)
        default: return nil
        }
    }

    private static func ensureSuccess(response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else {
            throw GatewayError(status: nil, message: "Studio returned a non-HTTP response.")
        }
        guard 200...299 ~= http.statusCode else {
            let value = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            let message = value?["error"] as? String ?? "Gateway request failed (\(http.statusCode))."
            throw GatewayError(status: http.statusCode, message: message)
        }
    }

    private static func path(_ value: String) -> String {
        value.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed.subtracting(CharacterSet(charactersIn: "/"))) ?? value
    }
}

private func certificateSha256(_ trust: SecTrust) -> String? {
    guard let chain = SecTrustCopyCertificateChain(trust) as? [SecCertificate],
          let certificate = chain.first else { return nil }
    let digest = SHA256.hash(data: SecCertificateCopyData(certificate) as Data)
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func answerPinnedChallenge(
    _ challenge: URLAuthenticationChallenge,
    fingerprint: String?,
    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
) {
    guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
          let fingerprint,
          let trust = challenge.protectionSpace.serverTrust else {
        completionHandler(.performDefaultHandling, nil)
        return
    }
    guard certificateSha256(trust) == fingerprint else {
        completionHandler(.cancelAuthenticationChallenge, nil)
        return
    }
    completionHandler(.useCredential, URLCredential(trust: trust))
}

private final class CertificatePinningDelegate: NSObject, URLSessionDelegate, @unchecked Sendable {
    private let fingerprint: String

    init(fingerprint: String) {
        self.fingerprint = fingerprint
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        answerPinnedChallenge(
            challenge,
            fingerprint: fingerprint,
            completionHandler: completionHandler
        )
    }
}

private final class UploadBridge: NSObject, URLSessionTaskDelegate, URLSessionDataDelegate, @unchecked Sendable {
    private let certificateSha256: String?
    private let onProgress: @Sendable (Int64, Int64) -> Void
    private var data = Data()
    private var response: URLResponse?
    private var completion: CheckedContinuation<(Data, URLResponse), Error>?
    private var session: URLSession?

    init(
        certificateSha256: String?,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void
    ) {
        self.certificateSha256 = certificateSha256
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        answerPinnedChallenge(
            challenge,
            fingerprint: certificateSha256,
            completionHandler: completionHandler
        )
    }

    func upload(request: URLRequest, fileURL: URL) async throws -> (Data, URLResponse) {
        try await withCheckedThrowingContinuation { continuation in
            completion = continuation
            let configuration = URLSessionConfiguration.ephemeral
            configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
            let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
            self.session = session
            session.uploadTask(with: request, fromFile: fileURL).resume()
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse, completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        self.response = response
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        self.data.append(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64, totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        onProgress(totalBytesSent, max(totalBytesExpectedToSend, 0))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        defer {
            completion = nil
            session.finishTasksAndInvalidate()
            self.session = nil
        }
        if let error {
            completion?.resume(throwing: error)
        } else if let response {
            completion?.resume(returning: (data, response))
        } else {
            completion?.resume(throwing: GatewayError(status: nil, message: "The learning upload ended without a response."))
        }
    }
}
