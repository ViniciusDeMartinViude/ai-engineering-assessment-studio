import { hashText } from "./context";

export interface CodeBlock {
  id: string;
  language: string;
  code: string;
  sha256: string;
}

const FENCED_BLOCK = /```([^\r\n`]*)\r?\n([\s\S]*?)(?:\r?\n)?```/g;

export function extractCodeBlocks(response: string): CodeBlock[] {
  const blocks: CodeBlock[] = [];
  let match: RegExpExecArray | null;
  while ((match = FENCED_BLOCK.exec(response)) !== null) {
    const code = match[2] ?? "";
    blocks.push({
      id: `block-${blocks.length + 1}`,
      language: (match[1] ?? "").trim().toLowerCase() || "text",
      code,
      sha256: hashText(code)
    });
  }
  return blocks;
}
