import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { AssessmentClient } from "../src/assessment/assessmentClient";
import { extractCodeBlocks } from "../src/ai/codeBlocks";
import { buildSelectedContext } from "../src/ai/context";
import { replacementTargetStillValid } from "../src/ai/editorGuards";
import { resolveConfigValues } from "../src/common/config";
import { buildMoveCommand, buildSuctionCommand, buildXyzCommand, canUseCartesian, RobotClient } from "../src/robot/robotClient";

test("extension manifest activates the AI Engineering views", () => {
  const manifest = JSON.parse(fs.readFileSync("package.json", "utf8"));
  assert.ok(manifest.activationEvents.includes("onView:aiEngineering.assessment"));
  assert.ok(manifest.activationEvents.includes("onView:aiEngineering.aiAssistant"));
  assert.ok(manifest.activationEvents.includes("onView:aiEngineering.robot"));
  assert.equal(manifest.contributes.viewsContainers.activitybar[0].title, "AI Engineering");
});

test("configuration uses safe URL settings and environment token", () => {
  const config = resolveConfigValues(
    {
      get<T>(key: string, fallback: T): T {
        return (key === "agentUrl" ? "http://127.0.0.1:8787/" : "") as T || fallback;
      }
    },
    { AI_GATEWAY_URL: "http://gateway.local/", AI_GATEWAY_SESSION_TOKEN: "practice-token" }
  );
  assert.equal(config.agentUrl, "http://127.0.0.1:8787");
  assert.equal(config.gatewayUrl, "http://gateway.local");
  assert.equal(config.gatewayToken, "practice-token");
});

test("assessment client calls versioned local agent routes", async () => {
  const calls: string[] = [];
  const client = new AssessmentClient(async (path) => {
    calls.push(path);
    return { ok: true } as never;
  });
  await client.status();
  await client.session();
  assert.deepEqual(calls, ["/api/v1/status", "/api/v1/session"]);
});

test("robot client builds existing JSON contracts", async () => {
  assert.deepEqual(buildMoveCommand("P1", "high", 1500), {
    command: "move",
    location: "P1",
    height: "high",
    duration_ms: 1500
  });
  assert.deepEqual(buildSuctionCommand("on"), { command: "suction", state: "on" });
  assert.deepEqual(buildXyzCommand(1, 2, 3, 1000), { command: "move_xyz", x: 1, y: 2, z: 3, duration_ms: 1000 });

  const payloads: unknown[] = [];
  const client = new RobotClient(async (_path, init) => {
    payloads.push(JSON.parse(String(init?.body)));
    return { ok: true, status: "accepted" } as never;
  });
  await client.command(buildMoveCommand("P2", "low", 500));
  assert.deepEqual(payloads[0], { command: "move", location: "P2", height: "low", duration_ms: 500 });
});

test("AI context sends only explicit selected text", () => {
  const context = buildSelectedContext({ selectedText: "print(1)", language: "python", source: "main.py" });
  assert.equal(context.text, "print(1)");
  assert.equal(context.source, "main.py");
  assert.equal(context.sha256.length, 64);
  assert.equal(JSON.stringify(context).includes("workspace"), false);
});

test("code blocks stay separate for controlled editor actions", () => {
  const blocks = extractCodeBlocks("a\n```python\nprint(1)\n```\n```ts\nconsole.log(2)\n```");
  assert.equal(blocks.length, 2);
  assert.equal(blocks[0].language, "python");
  assert.equal(blocks[1].code, "console.log(2)");
});

test("stale document versions or selections block replacement", () => {
  const snapshot = {
    uri: "file:///candidate/src/app.py",
    version: 7,
    range: { start: { line: 1, character: 0 }, end: { line: 1, character: 8 } },
    selectionHash: "hash-a"
  };
  assert.equal(replacementTargetStillValid(snapshot, { ...snapshot }), true);
  assert.equal(replacementTargetStillValid(snapshot, { ...snapshot, version: 8 }), false);
  assert.equal(replacementTargetStillValid(snapshot, { ...snapshot, selectionHash: "hash-b" }), false);
});

test("cartesian controls are unavailable unless backend reports capability", () => {
  assert.equal(canUseCartesian({ capabilities: { positions: true, suction: true, simulator: false, cartesian_move: false, software_stop: false } }), false);
  assert.equal(canUseCartesian({ capabilities: { positions: true, suction: true, simulator: true, cartesian_move: true, software_stop: false } }), true);
});
