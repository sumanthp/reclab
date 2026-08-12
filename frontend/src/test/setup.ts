import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Not using vitest's `globals: true`, so testing-library's auto-cleanup
// (which detects a global afterEach) doesn't kick in on its own — without
// this, DOM from one test leaks into the next within the same file.
afterEach(cleanup);
