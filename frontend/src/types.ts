// Mirrors api/main.py — the FastAPI wrapper around the tested
// earliest-arrival temporal router (src/routing/temporal_baseline.py).

export interface ContactIn {
  source_id: number;
  destination_id: number;
  start_s: number;
  end_s: number;
  data_rate_bps: number;
  propagation_delay_s: number;
  residual_capacity_bytes?: number;
}

export interface BundleIn {
  bundle_id: string;
  source_id: number;
  size_bytes: number;
  science_priority: number;
  deadline_s: number | null;
  data_type: string;
}

export interface RouteRequest {
  contacts: ContactIn[];
  bundle: BundleIn;
  destinations: number[];
}

export interface HopOut {
  source_id: number;
  destination_id: number;
  start_s: number;
  end_s: number;
  data_rate_bps: number;
  propagation_delay_s: number;
  available_at_s: number;
  depart_s: number;
  arrive_s: number;
  waited_s: number;
}

export type Verdict = "DELIVERED_ON_TIME" | "MISSED_DEADLINE" | "NO_FEASIBLE_ROUTE";

export interface RouteResponse {
  feasible: boolean;
  verdict: Verdict;
  path_ids: number[];
  path_names: string[];
  hops: HopOut[];
  arrival_s: number | null;
  slack_s: number | null;
  node_names: Record<string, string>;
  node_roles: Record<string, string>;
}

export interface Scenario {
  title: string;
  why: string;
  t_end: number;
  contacts: ContactIn[];
  bundle: BundleIn;
}

export type ScenarioMap = Record<string, Scenario>;
