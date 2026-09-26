# Omni AGI Companion for iPhone and iPad

This native SwiftUI client connects to the opt-in companion gateway in Omni AGI Studio. It does **not** embed, download, or fork a second model. The desktop and iOS client use the same brain identity, continuous chat, mutable neural state, learned tools, permissions, and audited actions.

## Pairing

1. Open **Phone companion** in the desktop Run surface.
2. Enable LAN access only on a trusted network when pairing a physical iPhone or iPad.
3. Enter one of the displayed addresses and the six-digit one-time code.
4. Revoke the device from Studio at any time.

The bearer credential is stored in the device Keychain. Studio stores only its SHA-256 digest. HTTP is restricted to loopback. Every LAN address is HTTPS and includes Studio's out-of-band SHA-256 certificate pin in its URL fragment; normal requests and streamed uploads both enforce that pin without sending it over the network.

The app supports continuous history, streamed chat, queueing while a turn is active, cancellation, and file/image/audio/video attachment learning. Attachments are uploaded from disk rather than encoded into JSON or copied into a mobile model.

## Build and test

Accept the installed Xcode license and select Xcode first:

```bash
sudo xcodebuild -license accept
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
npm run test:ios
npm run package:ios:simulator
```

CI boots an iPhone simulator, runs unit and UI tests against the same deterministic gateway protocol used by Android, and emits an unsigned IPA. An unsigned IPA is a build artifact for inspection or later signing; iOS will not install it directly.

## Custom-signed IPA

Apple requires an appropriate developer account, certificate, App ID, registered devices for development/ad-hoc distribution, and a matching provisioning profile. Automatic signing:

```bash
OMNI_IOS_TEAM_ID=YOURTEAMID \
OMNI_IOS_ALLOW_PROVISIONING_UPDATES=1 \
npm run package:ios:signed
```

Manual signing:

```bash
OMNI_IOS_TEAM_ID=YOURTEAMID \
OMNI_IOS_BUNDLE_ID=your.unique.bundle.id \
OMNI_IOS_SIGNING_IDENTITY="Apple Distribution: Your Name (YOURTEAMID)" \
OMNI_IOS_PROVISIONING_PROFILE="Your Ad Hoc Profile" \
npm run package:ios:signed
```

Artifacts are written beneath `release/mobile/`. Credentials and provisioning profiles are never added to the repository or `.omni` exports.
