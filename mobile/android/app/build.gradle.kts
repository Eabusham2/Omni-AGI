import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.Sync
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val repositoryRoot = rootProject.projectDir.parentFile.parentFile
val generatedComplianceAssets = layout.buildDirectory.dir("generated/omniComplianceAssets")
val prepareOmniComplianceAssets by tasks.registering(Sync::class) {
    into(generatedComplianceAssets.map { it.dir("legal") })
    from(repositoryRoot) {
        include("LICENSE.md", "COMMERCIAL_LICENSE.md", "THIRD_PARTY_NOTICES.md")
    }
    from(repositoryRoot.resolve("licenses")) {
        into("licenses")
        include("**/*")
    }
}

android {
    namespace = "ai.omniagi.companion"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.omniagi.companion"
        minSdk = 26
        targetSdk = 35
        versionCode = 10100
        versionName = "1.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            manifestPlaceholders["usesCleartextTraffic"] = "true"
        }
        release {
            isMinifyEnabled = false
            manifestPlaceholders["usesCleartextTraffic"] = "false"
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    sourceSets.getByName("main").assets.srcDir(generatedComplianceAssets)
    packaging {
        // Preserve dependency license resources. Omni's complete, canonical
        // notice set is also embedded under assets/legal for deterministic
        // verification without relying on dependency merge behavior.
        resources.excludes += "META-INF/DEPENDENCIES"
    }
}

tasks.named("preBuild").configure {
    dependsOn(prepareOmniComplianceAssets)
}

fun registerComplianceVerification(variant: String, apkName: String) =
    tasks.register<Exec>("verify${variant.replaceFirstChar { it.uppercase() }}OmniCompliance") {
        dependsOn("package${variant.replaceFirstChar { it.uppercase() }}")
        val artifact = layout.buildDirectory.file("outputs/apk/$variant/$apkName")
        doFirst {
            check(artifact.get().asFile.isFile) {
                "Android package is missing ${artifact.get().asFile}"
            }
        }
        commandLine(
            "node",
            repositoryRoot.resolve("scripts/verify-packaged-compliance.mjs"),
            "--repo-root",
            repositoryRoot,
            "--platform",
            "android",
            "--artifact",
            artifact.get().asFile,
        )
    }

val verifyDebugOmniCompliance = registerComplianceVerification("debug", "app-debug.apk")
val verifyReleaseOmniCompliance = registerComplianceVerification(
    "release",
    "app-release-unsigned.apk",
)
tasks.named("assembleDebug").configure { finalizedBy(verifyDebugOmniCompliance) }
tasks.named("assembleRelease").configure { finalizedBy(verifyReleaseOmniCompliance) }

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.16.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
