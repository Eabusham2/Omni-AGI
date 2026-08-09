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

/**
 * GitHub release assets do not preserve spaces in uploaded file names. Produce
 * the exact public name before checksums are written so downloaded files and
 * SHA256SUMS.txt always refer to the same path.
 */
export function publicReleaseAssetName(name) {
  const publicName = name.replace(/\s+/gu, ".");
  if (!/^[A-Za-z0-9._-]+$/u.test(publicName)) {
    throw new Error(`Release asset name contains unsupported characters: ${name}.`);
  }
  return publicName;
}
