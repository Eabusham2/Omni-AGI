/** Static existing-card presentation, not provider, IPC delivery or native learning proof. */
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { ApiTeacherPanel, teacherMediaOutputChoices } from "../src/renderer/src/ApiTeacherPanel";

it("offers only the typed media adapters on the existing Teacher card without any provider activity", () => {
  expect(teacherMediaOutputChoices("gemini")).toEqual(["image", "audio", "video"]);
  expect(teacherMediaOutputChoices("openai")).toEqual(["image"]);
  expect(teacherMediaOutputChoices("openrouter")).toEqual(["image"]);
  expect(teacherMediaOutputChoices("anthropic")).toEqual([]);
  expect(teacherMediaOutputChoices("ollama")).toEqual([]);
  const html = renderToStaticMarkup(<ApiTeacherPanel brainId="brain-a" initialProvider="gemini" onNotice={() => undefined} />);
  expect(html).toContain("Include generated media in these lessons");
  expect(html).toContain("Images"); expect(html).toContain("Audio"); expect(html).toContain("Video");
  expect(html).toContain("descriptions alone do not count");
  expect(html).not.toContain("checked=\"\"");
});
