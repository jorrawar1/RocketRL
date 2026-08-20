/** Single source of truth for colour, shared by CSS and the canvas layers. */
export const theme = {
  ground: "#0a0a0b",
  sunk: "#0e0e10",
  raised: "#121215",
  hairline: "#242428",
  hairlineHi: "#34343a",
  text: "#dfdfdf",
  dim: "#7a7a80",
  faint: "#4a4a50",
  blue: "#5b93eb",
  amber: "#fdb854",
  red: "#f1455b",

  display: '500 9px "Martian Mono Variable", "Martian Mono", ui-monospace, monospace',
  displaySm: '500 8px "Martian Mono Variable", "Martian Mono", ui-monospace, monospace',
  data: '400 11px "JetBrains Mono Variable", "JetBrains Mono", ui-monospace, monospace',
} as const;

export type Aesthetic = "minimal" | "airspace" | "optical" | "terminal";

export interface AestheticPalette {
  ground: string;
  sunk: string;
  raised: string;
  hairline: string;
  hairlineHi: string;
  text: string;
  dim: string;
  faint: string;
  blue: string;
  amber: string;
  red: string;
}

const palettes: Record<Aesthetic, AestheticPalette> = {
  minimal: theme,
  airspace: {
    ground: "#050506",
    sunk: "#08090c",
    raised: "#101116",
    hairline: "#282b32",
    hairlineHi: "#555b66",
    text: "#eceff4",
    dim: "#9299a5",
    faint: "#505660",
    blue: "#6aa7ff",
    amber: "#ffbf59",
    red: "#ff4962",
  },
  optical: {
    ground: "#05070b",
    sunk: "#070b12",
    raised: "#0c111b",
    hairline: "#1c2940",
    hairlineHi: "#34496e",
    text: "#e8f1ff",
    dim: "#7890ad",
    faint: "#3b4e68",
    blue: "#70b5ff",
    amber: "#ffb35b",
    red: "#ff5578",
  },
  terminal: {
    ground: "#050806",
    sunk: "#071009",
    raised: "#0b140d",
    hairline: "#18301d",
    hairlineHi: "#31583a",
    text: "#c7f4cf",
    dim: "#73a47d",
    faint: "#365d3e",
    blue: "#83e5a0",
    amber: "#f0cb66",
    red: "#ff6b65",
  },
};

export function paletteFor(aesthetic: Aesthetic): AestheticPalette {
  return palettes[aesthetic];
}

/** Hex plus alpha, so the palette stays the only place colours are written. */
export function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/** Colour that carries meaning for a mission phase. */
export function phaseColor(phase: string, aesthetic: Aesthetic = "minimal"): string {
  const palette = paletteFor(aesthetic);
  switch (phase) {
    case "SAMPLING":
      return palette.amber;
    case "FAILURE":
      return palette.red;
    case "OUTBOUND":
    case "RETURN":
    case "SUCCESS":
    default:
      return palette.blue;
  }
}
