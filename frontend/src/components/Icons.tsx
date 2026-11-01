import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function IconBase({ children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

export function ApertureIcon(props: IconProps) {
  return <IconBase {...props}><circle cx="12" cy="12" r="8.5" /><path d="m8 4.5 4 7.5M17.7 7H9M20 14h-8M16 19.5 12 12M6.3 17H15M4 10h8" /></IconBase>;
}

export function UploadIcon(props: IconProps) {
  return <IconBase {...props}><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" /><path d="M5 13v6h14v-6" /></IconBase>;
}

export function ImageIcon(props: IconProps) {
  return <IconBase {...props}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8.5" cy="9" r="1.5" /><path d="m4 17 5-5 3.5 3 2.5-2 5 4" /></IconBase>;
}

export function LayersIcon(props: IconProps) {
  return <IconBase {...props}><path d="m12 3-9 5 9 5 9-5-9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></IconBase>;
}

export function HeatIcon(props: IconProps) {
  return <IconBase {...props}><path d="M12 3v10.2a4 4 0 1 0 4 4c0-1.5-.8-2.8-2-3.5V5a2 2 0 0 0-4 0v8.7" /><path d="M12 17v-6" /></IconBase>;
}

export function DownloadIcon(props: IconProps) {
  return <IconBase {...props}><path d="M12 4v11m0 0 4-4m-4 4-4-4" /><path d="M5 19h14" /></IconBase>;
}

export function CheckIcon(props: IconProps) {
  return <IconBase {...props}><path d="m5 12 4 4L19 6" /></IconBase>;
}

export function AlertIcon(props: IconProps) {
  return <IconBase {...props}><path d="M12 3 2.8 20h18.4L12 3Z" /><path d="M12 9v4m0 3v.1" /></IconBase>;
}
