import XCTest

final class CompanionSmokeTests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testPairsWithSameBrainAndStreamsVisibleReply() throws {
        let configuredEndpoint = ProcessInfo.processInfo.environment["OMNI_GATEWAY_ENDPOINT"] ?? ""
        let endpoint = configuredEndpoint.isEmpty ? "http://127.0.0.1:41838" : configuredEndpoint
        let app = XCUIApplication()
        app.launchArguments = [
            "--reset-pairing",
            "--gateway-endpoint", endpoint,
            "--pair-code", "123456",
        ]
        app.launch()

        let pairButton = app.buttons["pairButton"]
        XCTAssertTrue(pairButton.waitForExistence(timeout: 8))
        pairButton.tap()

        let input = app.textFields["messageInput"]
        XCTAssertTrue(input.waitForExistence(timeout: 15))
        input.tap()
        input.typeText("Hello from the iOS simulator")
        app.buttons["sendButton"].tap()

        let reply = app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "Same-brain emulator reply")).firstMatch
        XCTAssertTrue(reply.waitForExistence(timeout: 15))
        let human = app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "Hello from the iOS simulator")).firstMatch
        let conversation = app.scrollViews["conversationScrollView"]
        XCTAssertTrue(conversation.exists)
        // Lazy rows can leave accessibility discovery when reply auto-follow
        // reaches the bottom with the keyboard open. Reveal only this actual
        // conversation; a lost or changed human message must still fail below.
        for _ in 0..<3 {
            if human.exists { break }
            conversation.swipeDown()
        }
        XCTAssertTrue(human.exists)
    }
}
