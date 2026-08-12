import {
  ApiError,
  type ArchitectureInfo,
  type ProfileResponse,
  type RunResponse,
  type RunSummary,
} from "./types";

async function handle<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // response body wasn't JSON — fall back to statusText
    }
    throw new ApiError(detail);
  }
  return (await resp.json()) as T;
}

export async function fetchArchitectures(): Promise<ArchitectureInfo[]> {
  return handle(await fetch("/architectures"));
}

export async function profileDataset(
  interactionsCsv: File,
  itemMetadataCsv: File | null,
  userCol: string,
  itemCol: string,
): Promise<ProfileResponse> {
  const form = new FormData();
  form.append("interactions_csv", interactionsCsv);
  if (itemMetadataCsv) form.append("item_metadata_csv", itemMetadataCsv);
  const params = new URLSearchParams({ user_col: userCol, item_col: itemCol });
  return handle(await fetch(`/profile?${params}`, { method: "POST", body: form }));
}

export async function startCompare(
  interactionsCsv: File,
  itemMetadataCsv: File | null,
  userCol: string,
  itemCol: string,
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("interactions_csv", interactionsCsv);
  if (itemMetadataCsv) form.append("item_metadata_csv", itemMetadataCsv);
  const params = new URLSearchParams({ user_col: userCol, item_col: itemCol });
  return handle(await fetch(`/compare?${params}`, { method: "POST", body: form }));
}

export async function getRun(jobId: string): Promise<RunResponse> {
  return handle(await fetch(`/runs/${jobId}`));
}

export async function listRuns(limit = 20): Promise<RunSummary[]> {
  return handle(await fetch(`/runs?limit=${limit}`));
}

export async function cancelRun(jobId: string): Promise<RunResponse> {
  return handle(await fetch(`/runs/${jobId}/cancel`, { method: "POST" }));
}
