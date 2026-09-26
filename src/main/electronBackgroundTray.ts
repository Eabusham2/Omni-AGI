import { existsSync } from "node:fs";
import {
  Menu,
  Tray,
  nativeImage,
  type MenuItemConstructorOptions,
  type NativeImage
} from "electron";
import type {
  BackgroundRuntimeCommands,
  BackgroundRuntimeTray,
  BackgroundRuntimeView
} from "./backgroundRuntimeController";

const FALLBACK_TRAY_SVG = [
  '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">',
  '<rect x="3" y="3" width="26" height="26" rx="8" fill="#17151f"/>',
  '<path d="M8 17h4l2-6 4 12 2-6h4" fill="none" stroke="#9b86ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>',
  "</svg>"
].join("");

function trayImage(iconPath?: string): NativeImage {
  let image = iconPath && existsSync(iconPath)
    ? nativeImage.createFromPath(iconPath)
    : nativeImage.createFromDataURL(
        `data:image/svg+xml;base64,${Buffer.from(FALLBACK_TRAY_SVG).toString("base64")}`
      );
  if (!image.isEmpty()) image = image.resize({ width: 20, height: 20 });
  if (process.platform === "darwin") image.setTemplateImage(true);
  return image;
}

export function backgroundTrayTemplate(
  view: BackgroundRuntimeView,
  commands: BackgroundRuntimeCommands
): MenuItemConstructorOptions[] {
  return [
    { label: view.title, enabled: false },
    { label: view.cognition.detail, enabled: false },
    { label: view.resourceStatus, enabled: false },
    { label: view.permissionStatus, enabled: false },
    { type: "separator" },
    {
      label: "Keep active after closing the window",
      type: "checkbox",
      checked: view.keepActiveAfterWindowClose,
      click: (item) => commands.setKeepActive(item.checked)
    },
    {
      label: "Open Omni AGI Studio",
      click: () => commands.openStudio()
    },
    {
      label: "Cancel background cognition",
      enabled: view.cancellable,
      click: () => commands.cancelBackground()
    },
    { type: "separator" },
    {
      label: "Quit Omni AGI Studio",
      click: () => commands.quitStudio()
    }
  ];
}

/** Native tray surface used while every Studio window is closed. */
export class ElectronBackgroundTray implements BackgroundRuntimeTray {
  private tray?: Tray;
  private lastCommands?: BackgroundRuntimeCommands;

  constructor(private readonly iconPath?: string) {}

  render(view: BackgroundRuntimeView, commands: BackgroundRuntimeCommands): void {
    this.lastCommands = commands;
    if (!this.tray) {
      this.tray = new Tray(trayImage(this.iconPath));
      this.tray.on("double-click", () => this.lastCommands?.openStudio());
    }
    this.tray.setToolTip(`Omni AGI Studio — ${view.title}`);
    this.tray.setContextMenu(
      Menu.buildFromTemplate(backgroundTrayTemplate(view, commands))
    );
  }

  destroy(): void {
    this.lastCommands = undefined;
    if (!this.tray) return;
    this.tray.destroy();
    this.tray = undefined;
  }
}
