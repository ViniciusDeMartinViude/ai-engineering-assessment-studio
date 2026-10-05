import * as vscode from "vscode";
import { RobotClient, RobotLocation, buildMoveCommand, buildSuctionCommand, buildXyzCommand, canUseCartesian } from "./robotClient";
import { RobotStatus } from "../common/types";

const LOCATIONS: RobotLocation[] = ["P1", "P2", "P3", "P4", "P5"];

export class RobotViewProvider implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  private status: RobotStatus | undefined;
  private lastCommandStatus = "No command sent from VS Code yet.";

  constructor(private readonly client: RobotClient) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.onDidReceiveMessage((message) => this.handleMessage(message));
    void this.refresh();
  }

  async refresh(): Promise<void> {
    try {
      this.status = await this.client.status();
    } catch (error) {
      this.status = undefined;
      this.lastCommandStatus = `Robot service unavailable: ${error instanceof Error ? error.message : String(error)}`;
    }
    this.render();
  }

  async health(): Promise<void> {
    await this.refresh();
    const health = this.status?.health;
    vscode.window.showInformationMessage(`Robot health: ${health?.state ?? "unknown"}`);
  }

  async refreshPositions(): Promise<void> {
    const positions = await this.client.positions();
    this.lastCommandStatus = `Positions refreshed: ${Object.keys((positions.positions as object | undefined) ?? {}).join(", ") || "none"}`;
    this.render();
  }

  async move(location: RobotLocation, height: "high" | "low", durationMs = 1500): Promise<void> {
    const result = await this.client.command(buildMoveCommand(location, height, durationMs));
    this.lastCommandStatus = `${location} ${height}: ${result.status}${result.error ? ` (${result.error})` : ""}`;
    this.render();
  }

  async suction(state: "on" | "off"): Promise<void> {
    const result = await this.client.command(buildSuctionCommand(state));
    this.lastCommandStatus = `Suction ${state}: ${result.status}${result.error ? ` (${result.error})` : ""}`;
    this.render();
  }

  async moveXyz(x: number, y: number, z: number, durationMs: number): Promise<void> {
    if (!canUseCartesian(this.status)) {
      vscode.window.showWarningMessage("XYZ movement is not reported as available by the assessment agent.");
      return;
    }
    const result = await this.client.command(buildXyzCommand(x, y, z, durationMs));
    this.lastCommandStatus = `XYZ move: ${result.status}${result.error ? ` (${result.error})` : ""}`;
    this.render();
  }

  private async handleMessage(message: { type?: string; location?: RobotLocation; height?: "high" | "low"; state?: "on" | "off"; x?: string; y?: string; z?: string; duration?: string }): Promise<void> {
    try {
      if (message.type === "refresh") {
        await this.refresh();
      } else if (message.type === "positions") {
        await this.refreshPositions();
      } else if (message.type === "move" && message.location && message.height) {
        await this.move(message.location, message.height);
      } else if (message.type === "suction" && message.state) {
        await this.suction(message.state);
      } else if (message.type === "xyz") {
        await this.moveXyz(Number(message.x), Number(message.y), Number(message.z), Number(message.duration || "1500"));
      }
    } catch (error) {
      this.lastCommandStatus = error instanceof Error ? error.message : String(error);
      this.render();
    }
  }

  private render(): void {
    if (!this.view) {
      return;
    }
    this.view.webview.html = this.html(this.view.webview);
  }

  private html(webview: vscode.Webview): string {
    const status = this.status;
    const canXyz = canUseCartesian(status);
    const rows = LOCATIONS.map((location) => `
      <div class="row">
        <strong>${location}</strong>
        <button data-type="move" data-location="${location}" data-height="high">High</button>
        <button data-type="move" data-location="${location}" data-height="low">Low</button>
      </div>`).join("");
    const nonce = String(Date.now());
    return `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 10px; }
    .kv { display: grid; grid-template-columns: 120px 1fr; gap: 6px 10px; margin-bottom: 12px; }
    .row { display: grid; grid-template-columns: 44px 1fr 1fr; gap: 6px; align-items: center; margin: 6px 0; }
    button { width: 100%; }
    input { width: 100%; box-sizing: border-box; }
    .xyz { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 10px; }
    .status { margin: 10px 0; padding: 8px; background: var(--vscode-editorWidget-background); }
    .muted { color: var(--vscode-descriptionForeground); }
  </style>
</head>
<body>
  <div class="kv">
    <span>Connection</span><strong>${escapeHtml(status?.health?.state ?? "unknown")}</strong>
    <span>Target</span><strong>${escapeHtml(status?.target_type ?? "unknown")}</strong>
    <span>Base URL</span><code>${escapeHtml(status?.base_url ?? "unavailable")}</code>
    <span>Suction</span><strong>${escapeHtml(status?.suction?.state ?? "unknown")}</strong>
    <span>Capabilities</span><span>${canXyz ? "XYZ available" : "XYZ unavailable"}</span>
  </div>
  <button data-type="refresh">Health</button>
  <button data-type="positions">Refresh positions</button>
  <div class="status">${escapeHtml(this.lastCommandStatus)}</div>
  ${rows}
  <div class="row">
    <strong>Suction</strong>
    <button data-type="suction" data-state="on">On</button>
    <button data-type="suction" data-state="off">Off</button>
  </div>
  <h3>XYZ</h3>
  ${canXyz ? `
    <div class="xyz">
      <input id="x" type="number" value="-3" aria-label="X">
      <input id="y" type="number" value="-130" aria-label="Y">
      <input id="z" type="number" value="89" aria-label="Z">
      <input id="duration" type="number" value="1500" aria-label="Duration">
    </div>
    <button id="moveXyz">Move XYZ</button>
  ` : `<p class="muted">The assessment agent has not reported cartesian_move=true for this target.</p>`}
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    document.querySelectorAll('button[data-type]').forEach((button) => {
      button.addEventListener('click', () => {
        vscode.postMessage({
          type: button.dataset.type,
          location: button.dataset.location,
          height: button.dataset.height,
          state: button.dataset.state
        });
      });
    });
    const xyz = document.getElementById('moveXyz');
    if (xyz) {
      xyz.addEventListener('click', () => vscode.postMessage({
        type: 'xyz',
        x: document.getElementById('x').value,
        y: document.getElementById('y').value,
        z: document.getElementById('z').value,
        duration: document.getElementById('duration').value
      }));
    }
  </script>
</body>
</html>`;
  }
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  }[char] ?? char));
}
