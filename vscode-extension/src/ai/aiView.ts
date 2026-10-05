import * as vscode from "vscode";
import { CodeBlock, extractCodeBlocks } from "./codeBlocks";
import { GatewayAllowance, GatewayResponse, SelectedAIContext } from "../common/types";
import { summarizeContext } from "./context";

export interface AiResultState {
  question: string;
  context: SelectedAIContext;
  responseText: string;
  raw: GatewayResponse;
  blocks: CodeBlock[];
}

export class AiStateStore {
  private result: AiResultState | undefined;

  set(question: string, context: SelectedAIContext, raw: GatewayResponse): AiResultState {
    const responseText = String(raw.answer ?? raw.response ?? "");
    this.result = {
      question,
      context,
      responseText,
      raw,
      blocks: extractCodeBlocks(responseText)
    };
    return this.result;
  }

  get(): AiResultState | undefined {
    return this.result;
  }
}

export class AIAssistantViewProvider implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  private allowance: GatewayAllowance | undefined;
  private message = "Select code and run an AI Engineering command, or ask a question here.";

  constructor(private readonly store: AiStateStore) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.onDidReceiveMessage((message) => {
      if (message.type === "ask" && typeof message.question === "string") {
        void vscode.commands.executeCommand("aiEngineering.ai.askFromView", message.question);
      } else if (message.type === "codeAction" && typeof message.action === "string" && typeof message.blockId === "string") {
        void vscode.commands.executeCommand(`aiEngineering.ai.${message.action}`, message.blockId);
      }
    });
    this.render();
  }

  setAllowance(allowance: GatewayAllowance | undefined): void {
    this.allowance = allowance;
    this.render();
  }

  setMessage(message: string): void {
    this.message = message;
    this.render();
  }

  showResult(): void {
    this.render();
  }

  private render(): void {
    if (!this.view) {
      return;
    }
    this.view.webview.html = this.html(this.view.webview);
  }

  private html(webview: vscode.Webview): string {
    const nonce = String(Date.now());
    const result = this.store.get();
    const model = String(this.allowance?.approved_model ?? this.allowance?.model ?? result?.raw.model ?? "unknown");
    const quota = this.allowance ? JSON.stringify(this.allowance) : "unavailable";
    const response = result?.responseText ?? "";
    const blockButtons = result?.blocks.map((block) => `
      <div class="block">
        <strong>${escapeHtml(block.id)} ${escapeHtml(block.language)}</strong>
        <button data-action="previewDiff" data-block="${block.id}">Preview Diff</button>
        <button data-action="insertAtCursor" data-block="${block.id}">Insert at Cursor</button>
        <button data-action="replaceSelection" data-block="${block.id}">Replace Selection</button>
        <button data-action="openUntitled" data-block="${block.id}">Open Untitled</button>
      </div>`).join("") ?? "";
    return `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 10px; }
    textarea { width: 100%; min-height: 72px; box-sizing: border-box; }
    button { margin-top: 6px; }
    .kv { display: grid; grid-template-columns: 110px 1fr; gap: 6px 10px; margin-bottom: 10px; }
    .status, .context { margin: 10px 0; padding: 8px; background: var(--vscode-editorWidget-background); }
    pre { white-space: pre-wrap; word-break: break-word; background: var(--vscode-textCodeBlock-background); padding: 8px; }
    .block { display: grid; grid-template-columns: 1fr repeat(4, auto); gap: 6px; align-items: center; margin: 6px 0; }
  </style>
</head>
<body>
  <div class="kv">
    <span>Model</span><strong>${escapeHtml(model)}</strong>
    <span>Allowance</span><code>${escapeHtml(quota)}</code>
  </div>
  <textarea id="question" placeholder="Ask the organizer gateway"></textarea>
  <button id="ask">Send</button>
  <div class="status">${escapeHtml(this.message)}</div>
  ${result ? `<div class="context">${escapeHtml(summarizeContext(result.context))}</div>` : ""}
  ${response ? `<h3>Response</h3><pre>${escapeHtml(response)}</pre>${blockButtons}` : ""}
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    document.getElementById('ask').addEventListener('click', () => {
      vscode.postMessage({ type: 'ask', question: document.getElementById('question').value });
    });
    document.querySelectorAll('button[data-action]').forEach((button) => {
      button.addEventListener('click', () => vscode.postMessage({
        type: 'codeAction',
        action: button.dataset.action,
        blockId: button.dataset.block
      }));
    });
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
