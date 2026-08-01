const PLATFORM_LABELS = {
  windows: "Windows",
  mac: "macOS",
  linux: "Linux"
};

/**
 * electron-builder exposes the target packager's native x64 architecture name
 * through ${arch}. AppImage uses x86_64, DEB uses amd64, and tar.gz keeps x64.
 */
export function packagedArchitecture(platform, architecture, extension) {
  if (platform === "linux" && architecture === "x64") {
    if (extension === "AppImage") return "x86_64";
    if (extension === "deb") return "amd64";
  }
  return architecture;
}

export function releaseArtifactName({
  product,
  version,
  platform,
  architecture,
  extension
}) {
  const platformLabel = PLATFORM_LABELS[platform];
  if (!platformLabel) {
    throw new Error(`Unsupported release platform ${String(platform)}.`);
  }
  const artifactArchitecture = packagedArchitecture(
    platform,
    architecture,
    extension
  );
  return `${product}-${version}-${platformLabel}-${artifactArchitecture}.${extension}`;
}
