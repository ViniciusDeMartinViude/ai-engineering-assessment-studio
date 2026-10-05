import { JsonRequester } from "../common/http";
import { RobotCommandResult, RobotStatus } from "../common/types";

export type RobotLocation = "P1" | "P2" | "P3" | "P4" | "P5";
export type RobotHeight = "high" | "low";
export type SuctionState = "on" | "off";

export interface RobotMoveCommand {
  command: "move";
  location: RobotLocation;
  height: RobotHeight;
  duration_ms: number;
}

export interface RobotSuctionCommand {
  command: "suction";
  state: SuctionState;
}

export interface RobotXyzCommand {
  command: "move_xyz";
  x: number;
  y: number;
  z: number;
  duration_ms: number;
}

export type RobotCommand = RobotMoveCommand | RobotSuctionCommand | RobotXyzCommand;

export function buildMoveCommand(location: RobotLocation, height: RobotHeight, durationMs: number): RobotMoveCommand {
  return { command: "move", location, height, duration_ms: durationMs };
}

export function buildSuctionCommand(state: SuctionState): RobotSuctionCommand {
  return { command: "suction", state };
}

export function buildXyzCommand(x: number, y: number, z: number, durationMs: number): RobotXyzCommand {
  return { command: "move_xyz", x, y, z, duration_ms: durationMs };
}

export function canUseCartesian(status: Pick<RobotStatus, "capabilities"> | undefined): boolean {
  return Boolean(status?.capabilities?.cartesian_move);
}

export class RobotClient {
  constructor(private readonly requestJson: JsonRequester) {}

  status(): Promise<RobotStatus> {
    return this.requestJson<RobotStatus>("/api/v1/robot/status");
  }

  positions(): Promise<Record<string, unknown>> {
    return this.requestJson<Record<string, unknown>>("/api/v1/robot/positions");
  }

  command(command: RobotCommand): Promise<RobotCommandResult> {
    return this.requestJson<RobotCommandResult>("/api/v1/robot/command", {
      method: "POST",
      body: JSON.stringify(command)
    });
  }
}
