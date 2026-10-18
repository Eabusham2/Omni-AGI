import { useEffect, useState } from "react";
import { Icon } from "./icons";

/** One submit action: Send while idle, Queue while busy, modifier+Steer. */
export function ComposerSubmitControl({
  active, stopping, canSend, canSteer, onSend, onSteer, onStop
}: {
  active: boolean;
  stopping: boolean;
  canSend: boolean;
  canSteer: boolean;
  onSend: () => void;
  onSteer: () => void;
  onStop: () => void;
}) {
  const [modifier, setModifier] = useState(false);
  useEffect(() => {
    const update = (event: Pick<KeyboardEvent, "ctrlKey" | "metaKey">) => setModifier(event.ctrlKey || event.metaKey);
    const clear = () => setModifier(false);
    window.addEventListener("keydown", update);
    window.addEventListener("keyup", update);
    window.addEventListener("pointermove", update, { passive: true });
    window.addEventListener("pointerdown", update, { passive: true });
    window.addEventListener("click", update, { passive: true });
    window.addEventListener("blur", clear);
    return () => {
      window.removeEventListener("keydown", update);
      window.removeEventListener("keyup", update);
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerdown", update);
      window.removeEventListener("click", update);
      window.removeEventListener("blur", clear);
    };
  }, []);
  const steering = active && canSteer && modifier;
  const label = active ? steering ? "Steer" : "Queue" : "Send";
  return (
    <div className="composer__turn-choices" role="group" aria-label="Message controls">
      {active ? (
        <button className="composer__stop-choice" type="button" onClick={onStop}
          disabled={stopping} aria-label={stopping ? "Stopping current turn" : "Stop current turn"}
          title="Stop responding; keep the conversation">
          <Icon name={stopping ? "pulse" : "pause"} size={14} />
          <span>{stopping ? "Stopping" : "Stop"}</span>
        </button>
      ) : null}
      <button className="send-button composer__submit" type="button"
        aria-label={steering ? "Steer current turn" : active ? "Queue message" : "Send message"}
        title={steering ? "Guide the current reply" : active ? "Queue message · hold Ctrl or ⌘ to steer" : "Send message"}
        disabled={!canSend || stopping}
        onClick={(event) => active && canSteer && (event.ctrlKey || event.metaKey) ? onSteer() : onSend()}>
        <Icon name={steering ? "arrow" : active ? "chat" : "send"} size={17} />
        <span>{label}</span>
      </button>
    </div>
  );
}
