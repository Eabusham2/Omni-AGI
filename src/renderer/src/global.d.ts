import type { OmniApi } from "@shared/types";

declare global {
  interface Window {
    omni?: OmniApi;
  }
}

export {};
