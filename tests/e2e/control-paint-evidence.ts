/** Observational audit evidence only; never product styles or host effects. */
export type AuditInkPaint = {
  target: string; kind: "text" | "graphic"; channel: string;
  color: string; background: string; opacity: string; contrast: number | null;
  contrastScope: "estimated-solid-composite" | "conservative-solid-ancestor-envelope" | "unmeasured";
  textLength?: number;
  diagnostics?: Record<string, unknown>;
};
export type ControlContentsPaint = {
  format: "visible-control-ink-v1"; complete: boolean;
  textRuns: AuditInkPaint[]; graphics: AuditInkPaint[]; unmeasured: string[];
  excludedDecorations?: Array<{ target: string; reason: "zero-area-empty-pseudo-background";
    width: number; height: number; boxShadow: string; contrastClaimed: false }>;
};
export type ControlPaintAdjudication = {
  examined: boolean; qualified: boolean; textMinimum: 4.5; graphicMinimum: 3;
  failures: string[];
};

/** The old successful/unmeasured parent path is deliberately unchanged. */
export function adjudicateLowParentPaint(parentContrast: number | null, contents?: ControlContentsPaint): ControlPaintAdjudication {
  const result: ControlPaintAdjudication = { examined: false, qualified: false, textMinimum: 4.5, graphicMinimum: 3, failures: [] };
  if (parentContrast === null || parentContrast >= 4.5) return result;
  result.examined = true;
  if (!Number.isFinite(parentContrast) || parentContrast < 1) result.failures.push("Invalid original parent contrast.");
  if (!contents || contents.format !== "visible-control-ink-v1") result.failures.push("Actual visible ink was not measured.");
  else {
    if (!contents.complete || contents.unmeasured.length) result.failures.push(...contents.unmeasured, "Visible ink coverage is incomplete.");
    const targets = [...contents.textRuns, ...contents.graphics];
    if (!targets.length) result.failures.push("No actual visible text or graphic ink was measured.");
    if (contents.textRuns.some(ink => ink.kind !== "text") || contents.graphics.some(ink => ink.kind !== "graphic"))
      result.failures.push("Visible ink kinds do not match their measured channels.");
    for (const ink of targets) {
      const required = ink.kind === "text" ? 4.5 : 3;
      const measured = ink.contrastScope === "estimated-solid-composite" || ink.kind === "graphic" && ink.contrastScope === "conservative-solid-ancestor-envelope";
      if (!measured || ink.contrast === null || !Number.isFinite(ink.contrast) || ink.contrast < required) {
        const reasons = ink.diagnostics?.unmeasuredReasons;
        result.failures.push(`${ink.target} ${ink.channel}: ${ink.contrast ?? "unmeasured"}:1; requires ${required}:1.${Array.isArray(reasons) && reasons.length ? ` ${reasons.join(" ")}` : ""}`);
      }
    }
  }
  result.qualified = result.failures.length === 0;
  return result;
}

/** Self-contained so Playwright can serialize this observational function. */
export function measureVisibleControlInk(node: Element): ControlContentsPaint {
  const doc = node.ownerDocument, view = doc.defaultView!;
  const evidence: ControlContentsPaint = { format: "visible-control-ink-v1", complete: false, textRuns: [], graphics: [], unmeasured: [], excludedDecorations: [] };
  const canvas = doc.createElement("canvas"); canvas.width = canvas.height = 1;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) { evidence.unmeasured.push("No solid-color canvas measurement."); return evidence; }
  const rgba = (color: string): number[] | null => {
    if (!view.CSS.supports("color", color)) return null;
    // CSS.supports alone cannot prove Canvas accepted the same syntax. An
    // unsupported setter must not silently reuse a previous dark ink value.
    context.fillStyle = "#010203"; context.fillStyle = color; const first = context.fillStyle;
    context.fillStyle = "#040506"; context.fillStyle = color;
    if (context.fillStyle !== first) return null;
    context.clearRect(0, 0, 1, 1); context.fillStyle = color; context.fillRect(0, 0, 1, 1);
    return [...context.getImageData(0, 0, 1, 1).data].map((value, index) => index === 3 ? value / 255 : value);
  };
  const over = (front: number[], back: number[]): number[] => {
    const alpha = front[3]! + back[3]! * (1 - front[3]!);
    return [0, 1, 2].map(index => alpha === 0 ? 0 : (front[index]! * front[3]! + back[index]! * back[3]! * (1 - front[3]!)) / alpha).concat(alpha);
  };
  const luminance = (rgb: number[]): number => rgb.slice(0, 3).map(value => {
    const unit = value / 255; return unit <= .04045 ? unit / 12.92 : ((unit + .055) / 1.055) ** 2.4;
  }).reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index]!, 0);
  const contrast = (ink: number[], background: number[]): number => {
    const fg = luminance(over(ink, background)), bg = luminance(background);
    return Math.round((Math.max(fg, bg) + .05) / (Math.min(fg, bg) + .05) * 1000) / 1000;
  };
  const describe = (target: Element): string => {
    const parts: string[] = [];
    for (let current: Element | null = target; current && current !== node; current = current.parentElement)
      parts.unshift(`${current.tagName.toLowerCase()}:nth-child(${current.parentElement ? [...current.parentElement.children].indexOf(current) + 1 : 1})`);
    return [":scope", ...parts].join(" > ");
  };
  const ancestors = (target: Element): Element[] => {
    const result: Element[] = [];
    for (let current: Element | null = target; current; current = current.parentElement) result.unshift(current);
    return result;
  };
  const visible = (target: Element, rectangles: ArrayLike<DOMRect>, expansion = 0): boolean => {
    let left = 0, right = view.innerWidth, top = 0, bottom = view.innerHeight;
    for (const ancestor of ancestors(target)) {
      const style = view.getComputedStyle(ancestor), box = ancestor.getBoundingClientRect();
      if (style.display === "none" || style.visibility !== "visible" || style.contentVisibility === "hidden" || Number(style.opacity) === 0) return false;
      if (ancestor !== target) {
        if (/hidden|clip|auto|scroll/.test(style.overflowX)) { left = Math.max(left, box.left); right = Math.min(right, box.right); }
        if (/hidden|clip|auto|scroll/.test(style.overflowY)) { top = Math.max(top, box.top); bottom = Math.min(bottom, box.bottom); }
      }
    }
    return [...Array.from(rectangles)].some(box => box.width + expansion * 2 > 0 && box.height + expansion * 2 > 0 &&
      box.right + expansion > left && box.left - expansion < right && box.bottom + expansion > top && box.top - expansion < bottom);
  };
  const surface = (target: Element) => {
    let background = [255, 255, 255, 1], opacity = 1, uncertainPaint = false, uncertainEffect = false, envelopeUncertain = false;
    const backgrounds: number[][] = [background];
    const layers: Array<{ target: string; color: string; image: string }> = [];
    for (const ancestor of ancestors(target)) {
      const style = view.getComputedStyle(ancestor), paint = rgba(style.backgroundColor);
      layers.push({ target: describe(ancestor), color: style.backgroundColor, image: style.backgroundImage });
      if (!paint) { uncertainPaint = true; envelopeUncertain = true; continue; }
      if (paint[3] === 1 && style.backgroundImage === "none") uncertainPaint = false;
      background = over(paint, background); backgrounds.push(background);
      opacity *= Number(style.opacity);
      uncertainPaint ||= style.backgroundImage !== "none";
      envelopeUncertain ||= style.backgroundImage !== "none";
      uncertainEffect ||= style.backdropFilter !== "none" || style.filter !== "none" || Number(style.opacity) !== 1 ||
        style.mixBlendMode !== "normal" || style.maskImage !== "none" || style.clipPath !== "none";
    }
    return { background, backgrounds, opacity, uncertain: uncertainPaint || uncertainEffect, envelopeUncertain, layers };
  };
  const inkPaint = (target: Element, kind: "text" | "graphic", channel: string, color: string, alpha = 1): AuditInkPaint => {
    if (/^currentcolor$/i.test(color)) color = view.getComputedStyle(target).color;
    const base = surface(target), ink = rgba(color);
    if (ink) ink[3] = ink[3]! * base.opacity * alpha;
    const uncertain = base.uncertain || !ink || !Number.isFinite(alpha);
    return { target: describe(target), kind, channel, color, background: `rgba(${base.background.join(",")})`, opacity: String(base.opacity * alpha),
      contrast: uncertain ? null : contrast(ink!, base.background), contrastScope: uncertain ? "unmeasured" : "estimated-solid-composite" };
  };
  const elements = [node, ...node.querySelectorAll("*")];
  for (const target of elements) {
    const style = view.getComputedStyle(target), tag = target.tagName.toLowerCase();
    const shape = target.namespaceURI === "http://www.w3.org/2000/svg" && /^(path|circle|ellipse|line|polyline|polygon|rect)$/.test(tag);
    const strokeWidth = Number.parseFloat(style.strokeWidth);
    if (!visible(target, shape ? [target.getBoundingClientRect()] : target.getClientRects(), shape && Number.isFinite(strokeWidth) ? strokeWidth / 2 : 0)) continue;
    if (/^(input|textarea|select|img|canvas|video|object|embed|iframe|image|use|foreignobject)$/.test(tag))
      evidence.unmeasured.push(`${describe(target)}: native/replaced content is not measured by text/SVG ink.`);
    for (const pseudo of ["::before", "::after"]) {
      const generated = view.getComputedStyle(target, pseudo), content = generated.content;
      if (["none", "normal"].includes(content) || generated.display === "none" || generated.visibility !== "visible" || Number(generated.opacity) === 0) continue;
      if (!["\"\"", "''"].includes(content)) { evidence.unmeasured.push(`${describe(target)}${pseudo}: generated content is unmeasured.`); continue; }
      const fill = rgba(/^currentcolor$/i.test(generated.backgroundColor) ? generated.color : generated.backgroundColor), borders = ["Top", "Right", "Bottom", "Left"].some(side =>
        Number.parseFloat(generated.getPropertyValue(`border-${side.toLowerCase()}-width`)) > 0 && !["none", "hidden"].includes(generated.getPropertyValue(`border-${side.toLowerCase()}-style`)));
      if (!fill || fill[3] === 0 && generated.backgroundImage === "none" && !borders && generated.boxShadow === "none") continue;
      const width = Number.parseFloat(generated.width), height = Number.parseFloat(generated.height), base = surface(target);
      const pixels = (value: string): number | null => /^-?\d+(?:\.\d+)?px$/.test(value) ? Number.parseFloat(value) : value === "0" ? 0 : null;
      const host = target.getBoundingClientRect(), left = pixels(generated.left), right = pixels(generated.right), top = pixels(generated.top), bottom = pixels(generated.bottom);
      const padX = [generated.paddingLeft, generated.paddingRight].map(pixels), padY = [generated.paddingTop, generated.paddingBottom].map(pixels);
      const boxWidth = width + (generated.boxSizing === "content-box" ? (padX[0] ?? NaN) + (padX[1] ?? NaN) : 0);
      const boxHeight = height + (generated.boxSizing === "content-box" ? (padY[0] ?? NaN) + (padY[1] ?? NaN) : 0);
      if ([boxWidth, boxHeight].every(Number.isFinite) && boxWidth >= 0 && boxHeight >= 0 && (boxWidth === 0 || boxHeight === 0) &&
          !borders && generated.outlineStyle === "none" && generated.backgroundImage === "none" && generated.filter === "none" &&
          generated.backdropFilter === "none" && generated.transform === "none" && generated.maskImage === "none" && generated.clipPath === "none") {
        // Empty zero-area fill cannot draw a glyph/line. A decorative glow
        // may still exist; record it without pretending to measure contrast.
        evidence.excludedDecorations!.push({ target: `${describe(target)}${pseudo}`, reason: "zero-area-empty-pseudo-background",
          width: boxWidth, height: boxHeight, boxShadow: generated.boxShadow, contrastClaimed: false });
        continue;
      }
      const x = left !== null ? host.left + (pixels(style.borderLeftWidth) ?? NaN) + left : right !== null ? host.right - (pixels(style.borderRightWidth) ?? NaN) - right - boxWidth : NaN;
      const y = top !== null ? host.top + (pixels(style.borderTopWidth) ?? NaN) + top : bottom !== null ? host.bottom - (pixels(style.borderBottomWidth) ?? NaN) - bottom - boxHeight : NaN;
      const footprintKnown = [x, y, boxWidth, boxHeight].every(Number.isFinite) && boxWidth > 0 && boxHeight > 0 &&
        [generated.marginLeft, generated.marginRight, generated.marginTop, generated.marginBottom].every(value => pixels(value) === 0) &&
        ancestors(target).every(value => view.getComputedStyle(value).transform === "none");
      const footprint = { left: x, right: x + boxWidth, top: y, bottom: y + boxHeight, width: boxWidth, height: boxHeight } as DOMRect;
      if (footprintKnown && !visible(target, [footprint])) continue;
      const family = ancestors(target);
      const probes = footprintKnown ? [[.1, .1], [.9, .1], [.5, .5], [.1, .9], [.9, .9]].map(([dx, dy]) => {
        const px = x + boxWidth * dx!, py = y + boxHeight * dy!, hit = doc.elementFromPoint(px, py);
        return { x: px, y: py, tag: hit?.tagName.toLowerCase() ?? null, inOwnerFamily: hit !== null && family.includes(hit) };
      }) : [];
      const unobscured = footprintKnown && probes.every(probe => probe.inOwnerFamily);
      let coverIndex = -1;
      if (footprintKnown) for (let index = 0; index < family.length; index++) {
        const ancestor = family[index]!, cover = view.getComputedStyle(ancestor), paint = rgba(cover.backgroundColor), box = ancestor.getBoundingClientRect();
        const square = [cover.borderTopLeftRadius, cover.borderTopRightRadius, cover.borderBottomLeftRadius, cover.borderBottomRightRadius]
          .every(value => value.split(/\s+/).every(part => Number.parseFloat(part) === 0));
        if (paint?.[3] === 1 && cover.backgroundImage === "none" && cover.backgroundClip === "border-box" && square &&
            cover.filter === "none" && cover.backdropFilter === "none" && cover.maskImage === "none" && cover.clipPath === "none" &&
            cover.mixBlendMode === "normal" && Number(cover.opacity) === 1 &&
            x >= box.left + (pixels(cover.borderLeftWidth) ?? Infinity) && x + boxWidth <= box.right - (pixels(cover.borderRightWidth) ?? Infinity) &&
            y >= box.top + (pixels(cover.borderTopWidth) ?? Infinity) && y + boxHeight <= box.bottom - (pixels(cover.borderBottomWidth) ?? Infinity)) coverIndex = index;
      }
      const envelope: number[][] = []; let envelopeUnknown = coverIndex < 0;
      if (coverIndex >= 0) {
        let backdrop = rgba(view.getComputedStyle(family[coverIndex]!).backgroundColor)!;
        envelope.push(backdrop);
        for (const ancestor of family.slice(coverIndex + 1)) {
          const layer = view.getComputedStyle(ancestor), paint = rgba(layer.backgroundColor);
          if (!paint || layer.backgroundImage !== "none") { envelopeUnknown = true; continue; }
          backdrop = over(paint, backdrop); envelope.push(backdrop);
        }
      }
      // A solid empty selection decoration can cross the control's edge.
      // Require3:1 against EVERY measured ancestor composite, not a guessed
      // backdrop chosen to make that decoration pass. Other pseudo paint is
      // explicitly unmeasured and keeps the original finding.
      const reasons = [
        !fill && "Pseudo background color was not decoded.",
        base.uncertain && "Actual solid parent paint has an unresolved image/effect.",
        coverIndex < 0 && "No exact opaque unfiltered ancestor covers the pseudo footprint.",
        envelopeUnknown && "Solid backdrop paint is unresolved inside the bounded opaque cover.",
        !footprintKnown && "Pseudo footprint is not bounded by resolved dimensions, offsets, margins and ancestor transforms.",
        !unobscured && "Pseudo footprint does not have measured unobscured owner/ancestor hits.",
        generated.position !== "absolute" && "Pseudo is not absolute-positioned.",
        style.position === "static" && "The host is not the known pseudo containing block.",
        (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) && "Pseudo width/height has no finite positive used value.",
        borders && "Pseudo border paint is unmeasured.",
        generated.backgroundImage !== "none" && "Pseudo background image is unmeasured.",
        generated.filter !== "none" && "Pseudo filter is unmeasured.",
        generated.backdropFilter !== "none" && "Pseudo backdrop filter is unmeasured.",
        generated.transform !== "none" && "Pseudo transform is unmeasured.",
        generated.maskImage !== "none" && "Pseudo mask is unmeasured.",
        generated.clipPath !== "none" && "Pseudo clipping path is unmeasured.",
        generated.mixBlendMode !== "normal" && "Pseudo blend mode is unmeasured.",
        Number(generated.opacity) !== 1 && "Pseudo group opacity is unmeasured."
      ].filter((value): value is string => typeof value === "string");
      const uncertain = reasons.length > 0;
      const paint: AuditInkPaint = { target: `${describe(target)}${pseudo}`, kind: "graphic", channel: "pseudo-background",
        color: generated.backgroundColor, background: envelope.map(value => `rgba(${value.join(",")})`).join(" | "),
        opacity: String(base.opacity), contrast: uncertain ? null : Math.min(...envelope.map(background => contrast(fill!, background))),
        contrastScope: uncertain ? "unmeasured" : "conservative-solid-ancestor-envelope",
        diagnostics: { unmeasuredReasons: reasons, content, position: generated.position, hostPosition: style.position,
          width: generated.width, height: generated.height, left: generated.left, right: generated.right, top: generated.top, bottom: generated.bottom,
          boxSizing: generated.boxSizing, margins: [generated.marginLeft, generated.marginRight, generated.marginTop, generated.marginBottom],
          footprintKnown, unobscured, footprint: { x, y, width: boxWidth, height: boxHeight }, probes,
          opaqueCover: coverIndex < 0 ? null : describe(family[coverIndex]!), ancestorPaints: base.layers } };
      evidence.graphics.push(paint);
    }
    if (shape) {
      if (target.closest("defs,symbol,clipPath,mask,marker,pattern")) continue;
      for (const channel of ["fill", "stroke"] as const) {
        const color = style[channel], alpha = Number(channel === "fill" ? style.fillOpacity : style.strokeOpacity);
        if (color === "none" || alpha === 0 || channel === "stroke" && strokeWidth === 0) continue;
        evidence.graphics.push(inkPaint(target, "graphic", `svg-${channel}`, color, alpha));
      }
    }
  }
  const walker = doc.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  for (let text = walker.nextNode(); text; text = walker.nextNode()) {
    if (!text.textContent?.trim() || !text.parentElement) continue;
    const target = text.parentElement, range = doc.createRange(); range.selectNodeContents(text);
    if (!visible(target, range.getClientRects())) continue;
    const style = view.getComputedStyle(target), svgText = target.namespaceURI === "http://www.w3.org/2000/svg";
    const paint = inkPaint(target, "text", svgText ? "svg-text-fill" : "text-fill", svgText ? style.fill : style.webkitTextFillColor || style.color);
    paint.textLength = text.textContent.length;
    if (style.textShadow !== "none" || Number.parseFloat(style.webkitTextStrokeWidth) > 0) {
      paint.contrast = null; paint.contrastScope = "unmeasured";
    }
    evidence.textRuns.push(paint);
  }
  evidence.complete = evidence.unmeasured.length === 0 && [...evidence.textRuns, ...evidence.graphics].every(paint => paint.contrast !== null);
  return evidence;
}
