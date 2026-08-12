import { describe, expect, it } from "vitest";
import { int, num, pct } from "./format";

describe("pct", () => {
  it("formats a fraction as a percentage with one decimal by default", () => {
    expect(pct(0.1414)).toBe("14.1%");
  });

  it("respects a custom digit count", () => {
    expect(pct(0.1414, 0)).toBe("14%");
  });

  it("handles values above 1 (e.g. an unrestricted candidate pool)", () => {
    expect(pct(1.528)).toBe("152.8%");
  });
});

describe("num", () => {
  it("formats with three decimals by default", () => {
    expect(num(0.1)).toBe("0.100");
  });
});

describe("int", () => {
  it("adds thousands separators", () => {
    expect(int(100000)).toBe("100,000");
  });
});
