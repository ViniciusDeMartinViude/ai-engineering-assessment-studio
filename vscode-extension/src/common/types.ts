export interface AssessmentStatus {
  ok: boolean;
  schema_version: string;
  workspace: string;
  candidate: string;
  session_id: string;
  mode: string;
  session_state: string;
  assessment_agent: {
    state: string;
    url?: string;
  };
  ai_gateway: {
    state: string;
    url?: string | null;
  };
  robot: RobotStatus;
}

export interface SessionStatus {
  ok: boolean;
  session: {
    candidate_id: string;
    session_id: string;
    mode: string;
    created_at: string;
    updated_at: string;
    schema_version: string;
  };
  workspace: string;
  pending_events: number;
}

export interface RobotCapabilities {
  positions: boolean;
  suction: boolean;
  simulator: boolean;
  cartesian_move: boolean;
  software_stop: boolean;
}

export interface RobotStatus {
  ok: boolean;
  target_type: "simulator" | "physical" | string;
  base_url: string;
  physical_armed: boolean;
  sequence_active: boolean;
  capabilities: RobotCapabilities;
  health: {
    state: string;
    error?: string;
    response?: RobotHttpResponse;
  };
  suction: {
    state: string;
    error?: string;
    response?: RobotHttpResponse;
  };
}

export interface RobotHttpResponse {
  method: string;
  path: string;
  status_code: number;
  elapsed_ms: number;
  data: Record<string, unknown>;
}

export interface RobotCommandResult {
  ok: boolean;
  status: "accepted" | "busy" | "uncertain" | "rejected" | "error" | string;
  note?: string;
  error?: string;
  response?: RobotHttpResponse;
}

export interface GatewayAllowance {
  approved_model?: string;
  model?: string;
  mode?: string;
  remaining_cents?: number;
  allowance_cents?: number;
  [key: string]: unknown;
}

export interface GatewayResponse {
  request_id?: string;
  status?: string;
  answer?: string;
  response?: string;
  model?: string;
  usage?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface SelectedAIContext {
  type: "selected_text" | "error_traceback" | "training_metrics";
  text: string;
  language?: string;
  source?: string;
  sha256: string;
}
