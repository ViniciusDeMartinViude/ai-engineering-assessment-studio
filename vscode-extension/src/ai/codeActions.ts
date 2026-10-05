import * as path from "path";
import * as vscode from "vscode";
import { AiClient } from "./aiClient";
import { AiStateStore, AIAssistantViewProvider } from "./aiView";
import { CodeBlock } from "./codeBlocks";
import { buildSelectedContext, hashText, summarizeContext } from "./context";
import { DocumentSnapshot, PlainRange, replacementTargetStillValid } from "./editorGuards";
import { SelectedAIContext } from "../common/types";

class StaticContentProvider implements vscode.TextDocumentContentProvider {
  private readonly changed = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.changed.event;
  private readonly content = new Map<string, string>();

  add(label: string, content: string): vscode.Uri {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const uri = vscode.Uri.parse(`ai-engineering-ai:/${encodeURIComponent(label)}?${id}`);
    this.content.set(uri.toString(), content);
    return uri;
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.content.get(uri.toString()) ?? "";
  }
}

export function registerAiCommands(
  context: vscode.ExtensionContext,
  getClient: () => Promise<AiClient | undefined>,
  view: AIAssistantViewProvider,
  store: AiStateStore
): void {
  const contentProvider = new StaticContentProvider();
  let replacementSnapshot: DocumentSnapshot | undefined;

  context.subscriptions.push(vscode.workspace.registerTextDocumentContentProvider("ai-engineering-ai", contentProvider));

  async function send(question: string, aiContext: SelectedAIContext, snapshot?: DocumentSnapshot): Promise<void> {
    const summary = summarizeContext(aiContext);
    const approval = await vscode.window.showInformationMessage(`Send this context to the organizer AI gateway? ${summary}`, { modal: true }, "Send");
    if (approval !== "Send") {
      return;
    }
    const client = await getClient();
    if (!client) {
      return;
    }
    replacementSnapshot = snapshot;
    view.setMessage("Waiting for organizer gateway response...");
    const response = await client.sendMessage(question, aiContext, `vscode-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    const result = store.set(question, aiContext, response);
    view.setMessage(`Received response with ${result.blocks.length} code block${result.blocks.length === 1 ? "" : "s"}.`);
    view.showResult();
    await vscode.commands.executeCommand("workbench.view.extension.aiEngineering");
  }

  function selectionContext(requireSelection: boolean): { context: SelectedAIContext; snapshot?: DocumentSnapshot } | undefined {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      if (requireSelection) {
        vscode.window.showWarningMessage("Open a file and select code before using this AI command.");
        return undefined;
      }
      return {
        context: buildSelectedContext({
          selectedText: "",
          source: "no editor"
        })
      };
    }
    const selectedText = editor.document.getText(editor.selection);
    if (requireSelection && !selectedText) {
      vscode.window.showWarningMessage("Select the code to send. The extension will not attach the whole file automatically.");
      return undefined;
    }
    const contextValue = buildSelectedContext({
      selectedText,
      language: editor.document.languageId,
      source: path.basename(editor.document.fileName || editor.document.uri.path)
    });
    const snapshot = selectedText ? {
      uri: editor.document.uri.toString(),
      version: editor.document.version,
      range: rangeToPlain(editor.selection),
      selectionHash: hashText(selectedText)
    } : undefined;
    return { context: contextValue, snapshot };
  }

  async function askWithSelection(question: string, requireSelection = true): Promise<void> {
    const built = selectionContext(requireSelection);
    if (!built) {
      return;
    }
    await send(question, built.context, built.snapshot);
  }

  async function chooseBlock(blockId?: string): Promise<CodeBlock | undefined> {
    const result = store.get();
    if (!result || result.blocks.length === 0) {
      vscode.window.showWarningMessage("The latest AI response does not contain a code block.");
      return undefined;
    }
    if (blockId) {
      return result.blocks.find((block) => block.id === blockId);
    }
    if (result.blocks.length === 1) {
      return result.blocks[0];
    }
    const picked = await vscode.window.showQuickPick(result.blocks.map((block) => ({
      label: `${block.id} ${block.language}`,
      description: block.sha256.slice(0, 12),
      block
    })));
    return picked?.block;
  }

  async function previewDiff(blockId?: string): Promise<void> {
    const block = await chooseBlock(blockId);
    if (!block) {
      return;
    }
    const editor = vscode.window.activeTextEditor;
    const original = editor ? editor.document.getText(editor.selection) : "";
    const left = contentProvider.add("Current Selection", original);
    const right = contentProvider.add(`AI ${block.id}.${block.language}`, block.code);
    await vscode.commands.executeCommand("vscode.diff", left, right, "AI Engineering Preview");
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("aiEngineering.ai.ask", async () => {
      const question = await vscode.window.showInputBox({ prompt: "Ask the organizer AI gateway" });
      if (question) {
        await askWithSelection(question, false);
      }
    }),
    vscode.commands.registerCommand("aiEngineering.ai.askFromView", async (question: string) => {
      if (question?.trim()) {
        await askWithSelection(question.trim(), false);
      }
    }),
    vscode.commands.registerCommand("aiEngineering.ai.explainSelection", () => askWithSelection("Explain the selected code.", true)),
    vscode.commands.registerCommand("aiEngineering.ai.fixSelection", () => askWithSelection("Fix the selected code. Return a concise explanation and a replacement code block.", true)),
    vscode.commands.registerCommand("aiEngineering.ai.improveSelection", () => askWithSelection("Improve the selected code while preserving its intent. Return a concise explanation and a replacement code block.", true)),
    vscode.commands.registerCommand("aiEngineering.ai.generateCode", async () => {
      const question = await vscode.window.showInputBox({ prompt: "Describe the code to generate" });
      if (question) {
        await askWithSelection(question, false);
      }
    }),
    vscode.commands.registerCommand("aiEngineering.ai.explainError", async () => {
      const text = await vscode.window.showInputBox({ prompt: "Paste the error or traceback to explain" });
      if (!text) {
        return;
      }
      await send("Explain this error and suggest a fix.", {
        type: "error_traceback",
        text,
        sha256: hashText(text)
      });
    }),
    vscode.commands.registerCommand("aiEngineering.ai.previewDiff", (blockId?: string) => previewDiff(blockId)),
    vscode.commands.registerCommand("aiEngineering.ai.insertAtCursor", async (blockId?: string) => {
      const block = await chooseBlock(blockId);
      const editor = vscode.window.activeTextEditor;
      if (!block || !editor) {
        return;
      }
      await editor.edit((edit) => edit.insert(editor.selection.active, block.code));
    }),
    vscode.commands.registerCommand("aiEngineering.ai.openUntitled", async (blockId?: string) => {
      const block = await chooseBlock(blockId);
      if (!block) {
        return;
      }
      const document = await vscode.workspace.openTextDocument({ language: block.language, content: block.code });
      await vscode.window.showTextDocument(document);
    }),
    vscode.commands.registerCommand("aiEngineering.ai.replaceSelection", async (blockId?: string) => {
      const block = await chooseBlock(blockId);
      const editor = vscode.window.activeTextEditor;
      if (!block || !editor) {
        return;
      }
      const current = {
        uri: editor.document.uri.toString(),
        version: editor.document.version,
        range: rangeToPlain(editor.selection),
        selectionHash: hashText(editor.document.getText(editor.selection))
      };
      if (!replacementTargetStillValid(replacementSnapshot, current)) {
        vscode.window.showWarningMessage("The document or selected range changed after the AI request. Choose a new destination before replacing code.");
        return;
      }
      await previewDiff(block.id);
      const approval = await vscode.window.showWarningMessage("Apply this AI replacement to the current unsaved editor buffer?", { modal: true }, "Apply");
      if (approval !== "Apply") {
        return;
      }
      await editor.edit((edit) => edit.replace(plainToRange(replacementSnapshot!.range), block.code));
    })
  );
}

function rangeToPlain(range: vscode.Range): PlainRange {
  return {
    start: { line: range.start.line, character: range.start.character },
    end: { line: range.end.line, character: range.end.character }
  };
}

function plainToRange(range: PlainRange): vscode.Range {
  return new vscode.Range(range.start.line, range.start.character, range.end.line, range.end.character);
}
