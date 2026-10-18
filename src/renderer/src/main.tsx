import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { RendererStartupGate } from "./RendererStartupGate";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RendererStartupGate bridge={window.omni} search={window.location.search} onReload={() => window.location.reload()}>
      <App />
    </RendererStartupGate>
  </React.StrictMode>
);
