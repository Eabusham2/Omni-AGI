import type { SVGProps } from "react";

export type IconName =
  | "activity"
  | "agents"
  | "archive"
  | "arrow"
  | "brain"
  | "chat"
  | "check"
  | "chevron"
  | "close"
  | "code"
  | "copy"
  | "database"
  | "download"
  | "expand"
  | "eye"
  | "file"
  | "fork"
  | "image"
  | "info"
  | "library"
  | "maximize"
  | "memory"
  | "minimize"
  | "more"
  | "pause"
  | "play"
  | "plus"
  | "pulse"
  | "search"
  | "send"
  | "settings"
  | "sparkles"
  | "terminal"
  | "trace"
  | "upload"
  | "video"
  | "volume"
  | "warning"
  | "wave";

interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

const paths: Record<IconName, React.ReactNode> = {
  activity: <path d="M3 12h3l2.2-5.3L12 18l2.4-6H21" />,
  agents: (
    <>
      <circle cx="9" cy="8" r="3" />
      <circle cx="17" cy="10" r="2.5" />
      <path d="M3.5 19c.5-4 2.3-6 5.5-6s5 2 5.5 6M14.5 14.5c3.5-.7 5.5.8 6 3.5" />
    </>
  ),
  archive: (
    <>
      <path d="M4 7h16v13H4zM3 4h18v3H3z" />
      <path d="M9 11h6" />
    </>
  ),
  arrow: <path d="m9 18 6-6-6-6" />,
  brain: (
    <>
      <path d="M9.5 4.5A3.5 3.5 0 0 0 6 8v.2A3.4 3.4 0 0 0 4 11a3.5 3.5 0 0 0 2.4 3.3A3.5 3.5 0 0 0 10 19V5.2" />
      <path d="M14.5 4.5A3.5 3.5 0 0 1 18 8v.2a3.4 3.4 0 0 1 2 2.8 3.5 3.5 0 0 1-2.4 3.3A3.5 3.5 0 0 1 14 19V5.2M7 9.5c1.8 0 3 1.1 3 2.8M17 9.5c-1.8 0-3 1.1-3 2.8" />
    </>
  ),
  chat: (
    <>
      <path d="M5 5h14v11H9l-4 3z" />
      <path d="M8 9h8M8 12h5" />
    </>
  ),
  check: <path d="m5 12 4 4L19 6" />,
  chevron: <path d="m8 10 4 4 4-4" />,
  close: <path d="m7 7 10 10M17 7 7 17" />,
  code: <path d="m9 18-6-6 6-6M15 6l6 6-6 6M14 4l-4 16" />,
  copy: (
    <>
      <rect x="8" y="8" width="11" height="11" rx="2" />
      <path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="7" ry="3" />
      <path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7" />
    </>
  ),
  download: <path d="M12 3v12m0 0 5-5m-5 5-5-5M5 20h14" />,
  expand: <path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />,
  eye: (
    <>
      <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
      <circle cx="12" cy="12" r="2.5" />
    </>
  ),
  file: (
    <>
      <path d="M6 3h8l4 4v14H6z" />
      <path d="M14 3v5h5M9 13h6M9 17h6" />
    </>
  ),
  fork: (
    <>
      <circle cx="6" cy="5" r="2" />
      <circle cx="18" cy="5" r="2" />
      <circle cx="12" cy="19" r="2" />
      <path d="M6 7v3c0 2 2 3 6 3s6-1 6-3V7M12 13v4" />
    </>
  ),
  image: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m4 17 5-5 4 4 3-3 5 5" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v6M12 7.5v.2" />
    </>
  ),
  library: (
    <>
      <path d="M4 5h4v15H4zM10 5h4v15h-4zM16 4l4 1.5-2 14.5-4-1.5z" />
    </>
  ),
  maximize: <rect x="5" y="5" width="14" height="14" />,
  memory: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 1v5M15 1v5M9 18v5M15 18v5M1 9h5M1 15h5M18 9h5M18 15h5M10 10h4v4h-4z" />
    </>
  ),
  minimize: <path d="M6 12h12" />,
  more: (
    <>
      <circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
      <circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" />
    </>
  ),
  pause: <path d="M8 5v14M16 5v14" />,
  play: <path d="m8 5 11 7-11 7z" />,
  plus: <path d="M12 5v14M5 12h14" />,
  pulse: <path d="M3 12h4l2-7 4 14 2-7h6" />,
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m16 16 5 5" />
    </>
  ),
  send: <path d="m3 4 18 8-18 8 3-8zm3 8h9" />,
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.8 1.8 0 0 0 .4 2l.1.1-2.8 2.8-.1-.1a1.8 1.8 0 0 0-2-.4 1.8 1.8 0 0 0-1.1 1.6v.2h-4V21A1.8 1.8 0 0 0 8.8 19.4a1.8 1.8 0 0 0-2 .4l-.1.1-2.8-2.8.1-.1a1.8 1.8 0 0 0 .4-2A1.8 1.8 0 0 0 2.8 14h-.2v-4h.2A1.8 1.8 0 0 0 4.4 9a1.8 1.8 0 0 0-.4-2l-.1-.1 2.8-2.8.1.1a1.8 1.8 0 0 0 2 .4A1.8 1.8 0 0 0 10 3V2.8h4V3a1.8 1.8 0 0 0 1.1 1.6 1.8 1.8 0 0 0 2-.4l.1-.1 2.8 2.8-.1.1a1.8 1.8 0 0 0-.4 2A1.8 1.8 0 0 0 21 10h.2v4H21a1.8 1.8 0 0 0-1.6 1Z" />
    </>
  ),
  sparkles: (
    <>
      <path d="m12 3 1.4 4.1L17.5 8l-4.1 1.4L12 13.5l-1.4-4.1L6.5 8l4.1-1.4zM19 14l.8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8zM5 14l.7 1.8 1.8.7-1.8.7L5 19l-.7-1.8-1.8-.7 1.8-.7z" />
    </>
  ),
  terminal: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="m7 9 3 3-3 3M12.5 16H17" />
    </>
  ),
  trace: (
    <>
      <path d="M5 4v16M5 7h7a3 3 0 0 1 3 3v4a3 3 0 0 0 3 3h1" />
      <circle cx="5" cy="6" r="2" />
      <circle cx="19" cy="17" r="2" />
    </>
  ),
  upload: <path d="M12 21V9m0 0 5 5m-5-5-5 5M5 4h14" />,
  video: (
    <>
      <rect x="3" y="6" width="13" height="12" rx="2" />
      <path d="m16 10 5-3v10l-5-3z" />
    </>
  ),
  volume: (
    <>
      <path d="M4 10v4h4l5 4V6l-5 4zM17 9c1.5 1.5 1.5 4.5 0 6M19.5 6.5c3 3 3 8 0 11" />
    </>
  ),
  warning: (
    <>
      <path d="m12 3 10 18H2z" />
      <path d="M12 9v5M12 17.5v.2" />
    </>
  ),
  wave: <path d="M3 12c2-7 4-7 6 0s4 7 6 0 4-7 6 0" />
};

export function Icon({ name, size = 18, ...props }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.65"
      {...props}
    >
      {paths[name]}
    </svg>
  );
}
