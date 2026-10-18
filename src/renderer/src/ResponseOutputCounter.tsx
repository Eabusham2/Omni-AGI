import type { PendingChatOutputPresentation } from "./chatLoadingPresentation";

export function ResponseOutputCounter({ presentation, cognitionWithoutOutput = false }: {
  presentation: PendingChatOutputPresentation; cognitionWithoutOutput?: boolean;
}) {
  return <span className={`response-token-counter response-token-counter--${presentation.phase}`}
    aria-label={presentation.ariaLabel} title={presentation.ariaLabel}>
    {cognitionWithoutOutput && presentation.outputTokens === 0 ? "No output tokens yet" : presentation.label}
  </span>;
}
