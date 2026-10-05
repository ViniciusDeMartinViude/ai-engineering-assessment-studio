import * as vscode from "vscode";
import { AssessmentClient } from "./assessmentClient";
import { AssessmentStatus } from "../common/types";

class AssessmentItem extends vscode.TreeItem {
  constructor(
    label: string,
    value: string,
    collapsibleState: vscode.TreeItemCollapsibleState = vscode.TreeItemCollapsibleState.None
  ) {
    super(`${label}: ${value}`, collapsibleState);
    this.tooltip = `${label}: ${value}`;
  }
}

export class AssessmentViewProvider implements vscode.TreeDataProvider<AssessmentItem> {
  private status: AssessmentStatus | undefined;
  private readonly changed = new vscode.EventEmitter<AssessmentItem | undefined | null | void>();
  readonly onDidChangeTreeData = this.changed.event;

  constructor(private readonly client: AssessmentClient) {}

  refresh(): void {
    void this.load();
  }

  async load(): Promise<void> {
    try {
      this.status = await this.client.status();
    } catch (error) {
      this.status = undefined;
      vscode.window.showWarningMessage(`AI Engineering assessment agent is unavailable: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      this.changed.fire();
    }
  }

  getTreeItem(element: AssessmentItem): vscode.TreeItem {
    return element;
  }

  getChildren(): AssessmentItem[] {
    if (!this.status) {
      return [
        new AssessmentItem("Assessment Agent", "Unavailable"),
        new AssessmentItem("Action", "Run AI Engineering: Refresh Assessment")
      ];
    }
    const robot = this.status.robot;
    return [
      new AssessmentItem("Candidate", this.status.candidate || "unknown"),
      new AssessmentItem("Mode", this.status.mode || "unknown"),
      new AssessmentItem("Session", this.status.session_state || "unknown"),
      new AssessmentItem("Workspace", this.status.workspace || "unknown"),
      new AssessmentItem("Assessment Agent", this.status.assessment_agent.state || "unknown"),
      new AssessmentItem("AI Gateway", this.status.ai_gateway.state || "unknown"),
      new AssessmentItem("Robot", `${robot.target_type || "unknown"} / ${robot.health?.state || "unknown"}`),
      new AssessmentItem("XYZ Capability", robot.capabilities?.cartesian_move ? "available" : "unavailable")
    ];
  }
}
