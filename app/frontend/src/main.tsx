/**
 * Phase 16.2.1 - the entry point.
 *
 * `StrictMode` is on. It double-invokes effects in development, which is not a nuisance here but a
 * test: a camera stream opened in an effect and not returned in its cleanup shows up immediately as
 * two tracks and a busy device, rather than in a bug report from a phone.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";

const host = document.getElementById("root");
if (!host) throw new Error("no #root in the document");

createRoot(host).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
