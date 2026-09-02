import { describe, expect, it } from "vitest";
import { formatValue, humanizeKey, metricHint } from "./formatters";

describe("formatters", () => {
  it("humanizes snake case keys", () => {
    expect(humanizeKey("champion_score")).toBe("Champion Score");
  });

  it("formats booleans for non-technical users", () => {
    expect(formatValue(true)).toBe("Yes");
    expect(formatValue(false)).toBe("No");
  });

  it("explains lower-is-better metrics", () => {
    expect(metricHint("rmse")).toBe("Lower is better");
  });
});