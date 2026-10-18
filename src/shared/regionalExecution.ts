import type { RegionalExecutionReview, RegionalExecutionStatus } from "./deviceResources";

export interface RegionalExecutionPrepared extends RegionalExecutionReview { reviewId: string }
export interface RegionalExecutionApi {
  prepare(request: { brainId: string; start?: number; count?: number }): Promise<RegionalExecutionPrepared>;
  status(brainId: string): Promise<RegionalExecutionStatus>;
  activate(request: { brainId: string; reviewId: string }): Promise<RegionalExecutionStatus>;
  deactivate(request: { brainId: string; epoch?: string }): Promise<RegionalExecutionStatus>;
}
