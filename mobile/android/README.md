# Omni AGI Android companion

This native Android app is a companion to the running Studio, not a second model. After an explicit pairing from **Device & runtime → Phone companion**, it uses the desktop process's existing `BrainService`, `ChatActionController`, neural worker, continuous conversation, tool permissions, and audit stream.

## Build

JDK 17 and Android SDK 35 are required.

```sh
./gradlew :app:testDebugUnitTest :app:assembleDebug
```

The installable, debug-signed APK is written to `app/build/outputs/apk/debug/app-debug.apk`. The root Android workflow also creates an API 35 emulator, connects through Android's `10.0.2.2` host route, pairs, streams a chat turn, and uploads a 2 MiB experience without buffering the whole source in the protocol.

## Pairing boundary

- Loopback/emulator access is the default.
- Physical-phone LAN access must be enabled explicitly in Studio. Protocol v2 advertises only HTTPS LAN addresses carrying Studio's SHA-256 certificate pin in the URL fragment.
- The Android client accepts cleartext HTTP only for loopback or the `10.0.2.2` emulator route. LAN HTTPS must include the exact Studio pin; redirects cannot downgrade it.
- The token is stored only in Android private preferences. Studio persists only its SHA-256 digest.
- A pairing never exports model weights, tool credentials, or an `.omni` bundle.
- Uploaded files stream to a temporary disk file, pass through the same complete dataset ingestion transaction, and are removed from staging afterward.
