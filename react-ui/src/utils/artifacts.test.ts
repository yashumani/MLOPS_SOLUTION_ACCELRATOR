import { describe, expect, it } from "vitest";
import { summarizeJsonArtifact } from "./artifacts";

describe("summarizeJsonArtifact", () => {
  it("promotes important fields into summary cards", () => {
    const summary = summarizeJsonArtifact("champion_manifest", {
      status: "completed",
      champion_phase: "phase_b",
      champion_score: 0.81,
      internal_blob: { raw: true }
    });

    expect(summary.fields.map((field) => field.label)).toContain("Champion Phase");
    expect(summary.fields.map((field) => field.label)).not.toContain("Internal Blob");
  });

  it("turns arrays of records into table rows", () => {
    const summary = summarizeJsonArtifact("leaderboard", [{ model: "xgboost", score: 0.7 }]);
    expect(summary.tableRows).toHaveLength(1);
    expect(summary.fields[0].value).toBe("1");
  });
});