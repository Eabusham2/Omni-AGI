import { execFile } from "node:child_process";
import {
  X509Certificate,
  createPrivateKey,
  generateKeyPairSync,
  randomBytes,
  randomUUID,
  sign
} from "node:crypto";
import { chmod, lstat, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";

const IDENTITY_FILE = "mobile-tls-identity.json";
const SIGNATURE_ALGORITHM_OID = "1.2.840.10045.4.3.2";
const execFileAsync = promisify(execFile);

// A Windows chmod(0600) does not remove inherited ACL grants. Keep the key
// under a protected, current-user-only DACL, including while it is staged.
const RESTRICT_WINDOWS_ACL = `
$ErrorActionPreference = 'Stop'
$path = $env:OMNI_MOBILE_TLS_IDENTITY_PATH
$attributes = [System.IO.File]::GetAttributes($path)
if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'TLS identity path cannot be a reparse point.' }
$isDirectory = [bool]($attributes -band [System.IO.FileAttributes]::Directory)
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = if ($isDirectory) { [System.IO.Directory]::GetAccessControl($path) } else { [System.IO.File]::GetAccessControl($path) }
$acl.SetAccessRuleProtection($true, $false)
foreach ($rule in @($acl.Access)) { [void]$acl.RemoveAccessRuleAll($rule) }
$inheritance = [System.Security.AccessControl.InheritanceFlags]::None
if ($isDirectory) {
  $inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
}
$grant = [System.Security.AccessControl.FileSystemAccessRule]::new($sid, [System.Security.AccessControl.FileSystemRights]::FullControl, $inheritance, [System.Security.AccessControl.PropagationFlags]::None, [System.Security.AccessControl.AccessControlType]::Allow)
$acl.AddAccessRule($grant)
$acl.SetOwner($sid)
if ($isDirectory) { [System.IO.Directory]::SetAccessControl($path, $acl) } else { [System.IO.File]::SetAccessControl($path, $acl) }
$verified = if ($isDirectory) { [System.IO.Directory]::GetAccessControl($path) } else { [System.IO.File]::GetAccessControl($path) }
$rules = @($verified.Access)
if (-not $verified.AreAccessRulesProtected -or $rules.Count -ne 1 -or $rules[0].AccessControlType -ne 'Allow' -or $rules[0].IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -ne $sid.Value -or $verified.GetOwner([System.Security.Principal.SecurityIdentifier]).Value -ne $sid.Value) {
  throw 'TLS identity ACL verification failed.'
}
`;

async function restrictWindowsIdentityAcl(path: string): Promise<void> {
  await execFileAsync("powershell.exe", [
    "-NoProfile", "-NonInteractive", "-Command", RESTRICT_WINDOWS_ACL
  ], {
    windowsHide: true,
    timeout: 30_000,
    env: { ...process.env, OMNI_MOBILE_TLS_IDENTITY_PATH: path }
  });
}

export interface MobileTlsIdentity {
  certificatePem: string;
  privateKeyPem: string;
  certificateSha256: string;
}

interface StoredMobileTlsIdentity extends MobileTlsIdentity {
  schemaVersion: 1;
  createdAt: string;
}

function lengthBytes(length: number): Buffer {
  if (!Number.isSafeInteger(length) || length < 0) throw new Error("Invalid DER length.");
  if (length < 0x80) return Buffer.from([length]);
  const bytes: number[] = [];
  for (let value = length; value > 0; value >>>= 8) bytes.unshift(value & 0xff);
  return Buffer.from([0x80 | bytes.length, ...bytes]);
}

function der(tag: number, ...values: Buffer[]): Buffer {
  const value = Buffer.concat(values);
  return Buffer.concat([Buffer.from([tag]), lengthBytes(value.length), value]);
}

const sequence = (...values: Buffer[]): Buffer => der(0x30, ...values);
const set = (...values: Buffer[]): Buffer => der(0x31, ...values);
const octetString = (value: Buffer): Buffer => der(0x04, value);
const utf8String = (value: string): Buffer => der(0x0c, Buffer.from(value, "utf8"));
const bitString = (value: Buffer, unusedBits = 0): Buffer =>
  der(0x03, Buffer.from([unusedBits]), value);

function integer(value: Buffer | number): Buffer {
  let bytes = typeof value === "number" ? Buffer.from([value]) : Buffer.from(value);
  while (bytes.length > 1 && bytes[0] === 0 && (bytes[1]! & 0x80) === 0) {
    bytes = bytes.subarray(1);
  }
  if ((bytes[0]! & 0x80) !== 0) bytes = Buffer.concat([Buffer.from([0]), bytes]);
  return der(0x02, bytes);
}

function base128(value: number): number[] {
  const output = [value & 0x7f];
  for (let remaining = Math.floor(value / 128); remaining > 0; remaining = Math.floor(remaining / 128)) {
    output.unshift((remaining & 0x7f) | 0x80);
  }
  return output;
}

function objectIdentifier(value: string): Buffer {
  const arcs = value.split(".").map(Number);
  if (
    arcs.length < 2 ||
    arcs.some((arc) => !Number.isSafeInteger(arc) || arc < 0) ||
    arcs[0]! > 2 ||
    (arcs[0]! < 2 && arcs[1]! > 39)
  ) throw new Error("Invalid certificate object identifier.");
  const bytes = [
    ...base128(arcs[0]! * 40 + arcs[1]!),
    ...arcs.slice(2).flatMap(base128)
  ];
  return der(0x06, Buffer.from(bytes));
}

function utcTime(value: Date): Buffer {
  const digits = value.toISOString().replace(/[-:T]/gu, "").slice(2, 14);
  return der(0x17, Buffer.from(`${digits}Z`, "ascii"));
}

function algorithmIdentifier(): Buffer {
  return sequence(objectIdentifier(SIGNATURE_ALGORITHM_OID));
}

function distinguishedName(): Buffer {
  return sequence(set(sequence(
    objectIdentifier("2.5.4.3"),
    utf8String("Omni AGI Studio Companion")
  )));
}

function extension(oid: string, value: Buffer, critical = false): Buffer {
  return sequence(
    objectIdentifier(oid),
    ...(critical ? [der(0x01, Buffer.from([0xff]))] : []),
    octetString(value)
  );
}

function ipv4Bytes(value: string): Buffer | undefined {
  const octets = value.split(".").map(Number);
  return octets.length === 4 && octets.every((octet) =>
    Number.isInteger(octet) && octet >= 0 && octet <= 255
  ) ? Buffer.from(octets) : undefined;
}

function certificateExtensions(ipAddresses: string[]): Buffer {
  const names = [der(0x82, Buffer.from("localhost", "ascii"))];
  for (const address of [...new Set(["127.0.0.1", "10.0.2.2", ...ipAddresses])]) {
    const bytes = ipv4Bytes(address);
    if (bytes) names.push(der(0x87, bytes));
  }
  return der(0xa3, sequence(
    extension("2.5.29.19", sequence(), true),
    extension("2.5.29.15", bitString(Buffer.from([0x80]), 7), true),
    extension(
      "2.5.29.37",
      sequence(objectIdentifier("1.3.6.1.5.5.7.3.1"))
    ),
    extension("2.5.29.17", sequence(...names))
  ));
}

function pem(label: string, value: Buffer): string {
  const encoded = value.toString("base64").match(/.{1,64}/gu)?.join("\n") ?? "";
  return `-----BEGIN ${label}-----\n${encoded}\n-----END ${label}-----\n`;
}

export function createMobileTlsIdentity(
  ipAddresses: string[],
  now = new Date()
): MobileTlsIdentity {
  const { privateKey, publicKey } = generateKeyPairSync("ec", {
    namedCurve: "prime256v1"
  });
  const serial = randomBytes(16);
  serial[0] = (serial[0] ?? 0) & 0x7f;
  if (serial.every((byte) => byte === 0)) serial[serial.length - 1] = 1;
  const notBefore = new Date(now.getTime() - 24 * 60 * 60_000);
  const notAfter = new Date(now);
  notAfter.setUTCFullYear(notAfter.getUTCFullYear() + 10);
  const tbs = sequence(
    der(0xa0, integer(2)),
    integer(serial),
    algorithmIdentifier(),
    distinguishedName(),
    sequence(utcTime(notBefore), utcTime(notAfter)),
    distinguishedName(),
    publicKey.export({ type: "spki", format: "der" }),
    certificateExtensions(ipAddresses)
  );
  const certificateDer = sequence(
    tbs,
    algorithmIdentifier(),
    bitString(sign("sha256", tbs, privateKey))
  );
  const certificate = new X509Certificate(certificateDer);
  return {
    certificatePem: pem("CERTIFICATE", certificateDer),
    privateKeyPem: privateKey.export({ type: "pkcs8", format: "pem" }).toString(),
    certificateSha256: certificate.fingerprint256.replaceAll(":", "").toLocaleLowerCase()
  };
}

function validatedIdentity(value: unknown): StoredMobileTlsIdentity {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("The mobile TLS identity is invalid.");
  }
  const record = value as Record<string, unknown>;
  if (
    record.schemaVersion !== 1 ||
    typeof record.createdAt !== "string" ||
    typeof record.certificatePem !== "string" ||
    typeof record.privateKeyPem !== "string" ||
    typeof record.certificateSha256 !== "string" ||
    !/^[a-f0-9]{64}$/u.test(record.certificateSha256)
  ) throw new Error("The mobile TLS identity is invalid.");
  const certificate = new X509Certificate(record.certificatePem);
  const privateKey = createPrivateKey(record.privateKeyPem);
  const fingerprint = certificate.fingerprint256.replaceAll(":", "").toLocaleLowerCase();
  if (
    fingerprint !== record.certificateSha256 ||
    !certificate.checkPrivateKey(privateKey) ||
    Date.parse(certificate.validTo) <= Date.now()
  ) throw new Error("The mobile TLS identity failed integrity validation.");
  return record as unknown as StoredMobileTlsIdentity;
}

export async function loadOrCreateMobileTlsIdentity(
  root: string,
  ipAddresses: string[]
): Promise<MobileTlsIdentity> {
  const path = join(root, IDENTITY_FILE);
  let exists = false;
  try {
    const entry = await lstat(path);
    if (!entry.isFile()) throw new Error("The mobile TLS identity path is not a regular file.");
    exists = true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  if (exists) {
    if (process.platform === "win32") await restrictWindowsIdentityAcl(path);
    const identity = validatedIdentity(JSON.parse(await readFile(path, "utf8")));
    if (process.platform !== "win32") await chmod(path, 0o600);
    return identity;
  }
  const identity = createMobileTlsIdentity(ipAddresses);
  const stored: StoredMobileTlsIdentity = {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    ...identity
  };
  await mkdir(root, { recursive: true });
  if (process.platform === "win32") {
    const stagingRoot = join(root, `.${IDENTITY_FILE}.${randomUUID()}.private`);
    await mkdir(stagingRoot);
    try {
      await restrictWindowsIdentityAcl(stagingRoot);
      const temporary = join(stagingRoot, IDENTITY_FILE);
      await writeFile(temporary, JSON.stringify(stored, null, 2), {
        encoding: "utf8",
        mode: 0o600,
        flag: "wx"
      });
      await restrictWindowsIdentityAcl(temporary);
      await rename(temporary, path);
      await restrictWindowsIdentityAcl(path);
    } finally {
      await rm(stagingRoot, { recursive: true, force: true });
    }
    return identity;
  }
  const temporary = `${path}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(stored, null, 2), {
    encoding: "utf8",
    mode: 0o600,
    flag: "wx"
  });
  await rename(temporary, path);
  return identity;
}
