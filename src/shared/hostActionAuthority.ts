/**
 * Main-process-only authority carried into a detached host action.
 *
 * The callback and AbortSignal are deliberately non-serializable. Persisted
 * receipts may retain their own ordinary evidence, but never this live grant.
 */
export interface HostActionOriginAuthority {
  /** Opaque exact-origin revision; never an approval token or credential. */
  revision: string;
  signal: AbortSignal;
  /** Revalidate immediately before every new effect boundary. */
  assertCurrent(): Promise<void>;
}
