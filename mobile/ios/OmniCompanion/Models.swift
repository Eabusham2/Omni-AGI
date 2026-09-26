import Foundation

struct PairedSession: Sendable {
    let endpoint: URL
    let token: String
    let deviceID: String
    let certificateSha256: String?
}

struct BrainSummary: Codable, Hashable, Identifiable, Sendable {
    let id: String
    let name: String
    let readiness: String
}

struct GatewayMessage: Codable, Hashable, Identifiable, Sendable {
    let id: String
    let role: String
    let content: String
    let createdAt: String
}

struct DisplayMessage: Identifiable, Equatable, Sendable {
    let id: String
    let role: String
    var content: String
    var pending: Bool
}

enum ChatFrame: Equatable, Sendable {
    case token(String)
    case state(String, String?)
    case action(String)
    case preview(String, Double?)
    case result(String)
    case failure(String, Bool)
}

struct UploadReceipt: Sendable {
    let fileName: String
    let bytes: Int64
    let sourceCount: Int
}

struct GatewayError: LocalizedError, Sendable {
    let status: Int?
    let message: String

    var errorDescription: String? { message }
}
