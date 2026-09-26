import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var model: CompanionViewModel

    var body: some View {
        Group {
            if model.isPaired { ChatView() }
            else { PairView() }
        }
        .tint(.indigo)
        .animation(.easeInOut(duration: 0.22), value: model.isPaired)
    }
}

private struct PairView: View {
    @EnvironmentObject private var model: CompanionViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    Spacer(minLength: 28)
                    ZStack {
                        Circle().fill(.indigo.gradient).frame(width: 78, height: 78)
                        Image(systemName: "brain.head.profile").font(.system(size: 34, weight: .semibold)).foregroundStyle(.white)
                    }
                    VStack(spacing: 8) {
                        Text("Omni AGI Companion").font(.largeTitle.bold()).multilineTextAlignment(.center)
                        Text("Pair this device to the persistent brain running in Omni AGI Studio. The model stays on your computer; this app shares its conversation, learning, and permitted actions.")
                            .font(.body).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: 580)

                    VStack(alignment: .leading, spacing: 14) {
                        Label("Companion address", systemImage: "desktopcomputer")
                            .font(.headline)
                        TextField("https://192.168.1.20:41837/#sha256=…", text: $model.endpointText)
                            .textInputAutocapitalization(.never)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                            .textFieldStyle(.roundedBorder)
                            .accessibilityIdentifier("gatewayEndpoint")
                        Label("One-time code", systemImage: "number.square")
                            .font(.headline)
                        TextField("123456", text: $model.pairingCode)
                            .keyboardType(.numberPad)
                            .textContentType(.oneTimeCode)
                            .textFieldStyle(.roundedBorder)
                            .accessibilityIdentifier("pairingCode")
                            .onChange(of: model.pairingCode) { value in
                                model.pairingCode = String(value.filter(\.isNumber).prefix(6))
                            }
                        Button(action: model.pair) {
                            HStack {
                                if model.isConnecting { ProgressView().tint(.white) }
                                Text(model.isConnecting ? "Pairing…" : "Pair this device").fontWeight(.semibold)
                            }
                            .frame(maxWidth: .infinity).padding(.vertical, 7)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isConnecting || model.pairingCode.count != 6)
                        .accessibilityIdentifier("pairButton")
                    }
                    .padding(20)
                    .frame(maxWidth: 620)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))

                    Label(model.status, systemImage: "info.circle")
                        .font(.footnote).foregroundStyle(.secondary).frame(maxWidth: 620, alignment: .leading)
                        .accessibilityIdentifier("connectionStatus")
                    Text("LAN addresses use Studio's certificate-pinned TLS identity. Plain HTTP is accepted only for loopback development.")
                        .font(.caption).foregroundStyle(.secondary).frame(maxWidth: 620)
                    Spacer(minLength: 30)
                }
                .padding(.horizontal, 22)
            }
            .background(AdaptiveBackground())
        }
    }
}

private struct ChatView: View {
    @EnvironmentObject private var model: CompanionViewModel
    @State private var showImporter = false

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if model.messages.isEmpty {
                            VStack(spacing: 12) {
                                Image(systemName: "brain.head.profile").font(.system(size: 42)).foregroundStyle(.secondary)
                                Text("Continuous conversation").font(.title3.bold())
                                Text("This is the same chat and mutable neural state as the desktop brain.")
                                    .foregroundStyle(.secondary).multilineTextAlignment(.center)
                            }
                            .padding(.top, 90)
                        }
                        ForEach(model.messages) { message in
                            MessageBubble(message: message).id(message.id)
                        }
                    }
                    .padding(.horizontal, 14).padding(.vertical, 18)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: model.messages) { _ in
                    guard let id = model.messages.last?.id else { return }
                    withAnimation(.easeOut(duration: 0.18)) { proxy.scrollTo(id, anchor: .bottom) }
                }
                .safeAreaInset(edge: .bottom) { Composer(showImporter: $showImporter) }
            }
            .background(AdaptiveBackground())
            .navigationTitle(model.selectedBrainName)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Menu {
                        Picker("Brain", selection: $model.selectedBrainID) {
                            ForEach(model.brains) { brain in
                                Text(brain.readiness == "ready" ? brain.name : "\(brain.name) · \(brain.readiness)").tag(brain.id)
                            }
                        }
                        Button("Refresh brains", systemImage: "arrow.clockwise", action: model.refreshBrains)
                    } label: {
                        Label("Brain", systemImage: "brain")
                    }
                    .accessibilityIdentifier("brainMenu")
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Text("Same persistent brain")
                        Divider()
                        Button("Disconnect this device", systemImage: "iphone.slash", role: .destructive, action: model.disconnect)
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .overlay(alignment: .top) {
                StatusPill().padding(.top, 8)
            }
            .fileImporter(isPresented: $showImporter, allowedContentTypes: [.item], allowsMultipleSelection: true) { result in
                if case .success(let files) = result { model.learn(files: files) }
            }
        }
    }
}

private struct MessageBubble: View {
    let message: DisplayMessage

    private var isHuman: Bool { message.role == "human" }

    var body: some View {
        HStack {
            if isHuman { Spacer(minLength: 48) }
            VStack(alignment: .leading, spacing: 5) {
                Text(isHuman ? "You" : message.role == "brain" ? "Brain" : message.role.capitalized)
                    .font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                if message.content.isEmpty && message.pending {
                    HStack(spacing: 5) {
                        ForEach(0..<3, id: \.self) { index in
                            Circle().frame(width: 6, height: 6).opacity(0.45 + Double(index) * 0.2)
                        }
                    }
                    .padding(.vertical, 5)
                    .accessibilityLabel("Brain is responding")
                } else {
                    Text(message.content).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 11)
            .background(isHuman ? Color.indigo.opacity(0.16) : Color.secondary.opacity(0.10), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .frame(maxWidth: 720, alignment: isHuman ? .trailing : .leading)
            if !isHuman { Spacer(minLength: 48) }
        }
        .accessibilityIdentifier(isHuman ? "humanMessage" : "brainMessage")
    }
}

private struct Composer: View {
    @EnvironmentObject private var model: CompanionViewModel
    @Binding var showImporter: Bool

    var body: some View {
        VStack(spacing: 8) {
            if model.isLearning {
                HStack {
                    ProgressView(value: model.learningProgress)
                    Text(model.learningProgress.map { "\(Int($0 * 100))%" } ?? "Learning…").font(.caption.monospacedDigit())
                }
            }
            HStack(alignment: .bottom, spacing: 8) {
                Button { showImporter = true } label: {
                    Image(systemName: "paperclip").frame(width: 32, height: 32)
                }
                .buttonStyle(.bordered)
                .disabled(model.isLearning)
                .accessibilityIdentifier("attachButton")
                TextField(model.isChatting ? "Queue another message" : "Message this brain", text: $model.draft, axis: .vertical)
                    .lineLimit(1...6)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 12).padding(.vertical, 10)
                    .background(Color.secondary.opacity(0.10), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .submitLabel(.send)
                    .onSubmit(model.sendOrQueue)
                    .accessibilityIdentifier("messageInput")
                if model.isChatting {
                    Button(action: model.stopTurn) { Image(systemName: "stop.fill").frame(width: 32, height: 32) }
                        .buttonStyle(.borderedProminent).tint(.red)
                        .accessibilityIdentifier("stopButton")
                } else {
                    Button(action: model.sendOrQueue) { Image(systemName: "arrow.up").frame(width: 32, height: 32) }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .accessibilityIdentifier("sendButton")
                }
            }
            if model.queuedCount > 0 {
                Text("\(model.queuedCount) queued").font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 4)
        .background(.ultraThinMaterial)
    }
}

private struct StatusPill: View {
    @EnvironmentObject private var model: CompanionViewModel

    var body: some View {
        Label(model.status, systemImage: model.isChatting ? "waveform.path.ecg" : model.isLearning ? "brain.fill" : "checkmark.circle.fill")
            .font(.caption).lineLimit(2)
            .padding(.horizontal, 12).padding(.vertical, 7)
            .background(.thinMaterial, in: Capsule())
            .shadow(color: .black.opacity(0.08), radius: 8, y: 3)
            .padding(.horizontal, 18)
            .accessibilityIdentifier("runtimeStatus")
    }
}

private struct AdaptiveBackground: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        LinearGradient(
            colors: scheme == .dark
                ? [Color(red: 0.045, green: 0.055, blue: 0.10), Color(red: 0.09, green: 0.06, blue: 0.15)]
                : [Color(red: 0.96, green: 0.97, blue: 1), Color(red: 0.98, green: 0.95, blue: 1)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
}
