export interface McpOAuthStart {
  clientId?: string;
  clientSecret?: string;
  clientMetadataUrl?: string;
  issuer?: string;
  redirectPort?: number;
}
export interface McpOAuthStatus {
  state: "none" | "required" | "authorizing" | "authorized" | "error";
  flowId?: string;
  issuer?: string;
  scopes?: string[];
  persistence?: "encrypted" | "session";
  error?: string;
}
export interface McpCredentialUpdate {
  /** Empty removes the HTTP credential and any prior OAuth login. */
  bearerToken?: string;
  /** Replaces the complete explicit stdio environment. Empty clears it. */
  environment?: Record<string, string>;
}
