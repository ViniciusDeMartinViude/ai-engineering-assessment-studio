import * as vscode from "vscode";
import { RobotViewProvider } from "./robotView";
import { RobotLocation } from "./robotClient";

const LOCATIONS: RobotLocation[] = ["P1", "P2", "P3", "P4", "P5"];

export function registerRobotCommands(context: vscode.ExtensionContext, provider: RobotViewProvider): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("aiEngineering.robot.openControl", async () => {
      await vscode.commands.executeCommand("workbench.view.extension.aiEngineering");
      await provider.refresh();
    }),
    vscode.commands.registerCommand("aiEngineering.robot.health", () => provider.health()),
    vscode.commands.registerCommand("aiEngineering.robot.refreshPositions", () => provider.refreshPositions()),
    vscode.commands.registerCommand("aiEngineering.robot.suctionOn", () => provider.suction("on")),
    vscode.commands.registerCommand("aiEngineering.robot.suctionOff", () => provider.suction("off"))
  );

  for (const location of LOCATIONS) {
    for (const height of ["high", "low"] as const) {
      const command = `aiEngineering.robot.move${location}${height === "high" ? "High" : "Low"}`;
      context.subscriptions.push(
        vscode.commands.registerCommand(command, () => provider.move(location, height))
      );
    }
  }
}
