import type { RouteRequest, RouteResponse, ScenarioMap } from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<TRes>;
}

export function fetchScenarios(): Promise<ScenarioMap> {
  return get<ScenarioMap>("/scenarios");
}

export function computeRoute(req: RouteRequest): Promise<RouteResponse> {
  return post<RouteRequest, RouteResponse>("/route", req);
}
