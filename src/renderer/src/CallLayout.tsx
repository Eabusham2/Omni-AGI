import type { ReactNode } from "react";
import "./callLayout.css";

/** Hiding controls does not stop a call or unmount its real capture owner. */
export function CallLayout({ open, controls, children }: { open: boolean; controls: ReactNode; children: ReactNode }) {
  return <div className={`call-layout${open ? " is-open" : ""}`}>
    {controls}<div className="call-layout__conversation">{children}</div>
  </div>;
}
