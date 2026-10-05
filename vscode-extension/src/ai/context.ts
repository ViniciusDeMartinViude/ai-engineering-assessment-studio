import { createHash } from "crypto";
import { SelectedAIContext } from "../common/types";

export interface EditorSelectionInput {
  selectedText: string;
  language?: string;
  source?: string;
}

export function hashText(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

export function buildSelectedContext(input: EditorSelectionInput): SelectedAIContext {
  return {
    type: "selected_text",
    text: input.selectedText,
    language: input.language,
    source: input.source,
    sha256: hashText(input.selectedText)
  };
}

export function summarizeContext(context: SelectedAIContext): string {
  const lines = context.text.split(/\r?\n/).length;
  const source = context.source ? ` from ${context.source}` : "";
  return `${context.type}${source}: ${context.text.length} characters, ${lines} line${lines === 1 ? "" : "s"}, sha256 ${context.sha256.slice(0, 12)}`;
}
