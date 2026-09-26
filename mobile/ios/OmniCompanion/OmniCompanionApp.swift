import SwiftUI

@main
struct OmniCompanionApp: App {
    @StateObject private var model = CompanionViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
        }
    }
}
