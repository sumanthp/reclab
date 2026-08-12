import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const dirname = path.dirname(fileURLToPath(import.meta.url));
const INTERACTIONS_CSV = path.join(dirname, "fixtures", "interactions.csv");
const ITEM_METADATA_CSV = path.join(dirname, "fixtures", "item_metadata.csv");

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("uploads a dataset and sees the profile and reasoning engine shortlist", async ({
  page,
}) => {
  await page.getByLabel("Interactions CSV (required)").setInputFiles(INTERACTIONS_CSV);
  await page.getByLabel("Item metadata CSV (optional)").setInputFiles(ITEM_METADATA_CSV);
  await page.getByRole("button", { name: "Profile dataset" }).click();

  await expect(page.getByRole("heading", { name: "Data profile" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Reasoning engine shortlist" })).toBeVisible();

  // All three registered architectures should appear as their own shortlist
  // card — this is the real /profile response, not a fixture. Scoped to
  // .arch-name (not a free-text search) since "sasrec" also appears inside
  // hybrid_llm's collapsed description ("SASRec-style encoder...").
  for (const architecture of ["two_tower", "sasrec", "hybrid_llm"]) {
    await expect(page.locator(".arch-name", { hasText: architecture })).toBeVisible();
  }
});

test("runs a full comparison end to end and it appears in run history", async ({ page }) => {
  await page.getByLabel("Interactions CSV (required)").setInputFiles(INTERACTIONS_CSV);
  await page.getByLabel("Item metadata CSV (optional)").setInputFiles(ITEM_METADATA_CSV);
  await page.getByRole("button", { name: "Profile dataset" }).click();
  await expect(page.getByRole("heading", { name: "Compare" })).toBeVisible();

  await page.getByRole("button", { name: "Run full comparison" }).click();

  // Real training + eval of all three architectures on the tiny fixture
  // dataset — fast, but polled over the network, so give it real headroom
  // rather than asserting on a fixed sleep.
  await expect(page.getByText("Download results (JSON)")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/example recommendations/i).first()).toBeVisible();

  // The run should now show up in the history rail, and reselecting it
  // should restore the same completed comparison view.
  const historyItem = page.locator(".run-history-item").first();
  await expect(historyItem).toContainText("Done");
  await historyItem.click();
  await expect(page.getByText("Download results (JSON)")).toBeVisible();
});
