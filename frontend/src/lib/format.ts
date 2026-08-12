export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number, digits = 3): string {
  return value.toFixed(digits);
}

export function int(value: number): string {
  return value.toLocaleString();
}
