import * as vscode from "vscode";
import { AssessmentClient } from "./assessment/assessmentClient";
import { AssessmentViewProvider } from "./assessment/assessmentView";
import { AiClient } from "./ai/aiClient";
import { registerAiCommands } from "./ai/codeActions";
import { AiStateStore, AIAssistantViewProvider } from "./ai/aiView";
import { createJsonRequester } from "./common/http";
import { resolveConfigValues } from "./common/config";
import { RobotClient } from "./robot/robotClient";
import { registerRobotCommands } from "./robot/robotCommands";
import { RobotViewProvider } from "./robot/robotView";

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const config = currentConfig();
  const assessmentClient = new AssessmentClient(createJsonRequester(config.agentUrl));
  const assessmentProvider = new AssessmentViewProvider(assessmentClient);
  const robotProvider = new RobotViewProvider(new RobotClient(createJsonRequester(config.agentUrl)));
  const aiStore = new AiStateStore();
  const aiProvider = new AIAssistantViewProvider(aiStore);

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("aiEngineering.assessment", assessmentProvider),
    vscode.window.registerWebviewViewProvider("aiEngineering.robot", robotProvider),
    vscode.window.registerWebviewViewProvider("aiEngineering.aiAssistant", aiProvider),
    vscode.commands.registerCommand("aiEngineering.refreshAssessment", () => assessmentProvider.refresh())
  );

  registerRobotCommands(context, robotProvider);
  registerAiCommands(context, () => getAiClient(false), aiProvider, aiStore);

  await assessmentProvider.load();
  await refreshAllowance();

  async function getAiClient(silent: boolean): Promise<AiClient | undefined> {
    const latest = currentConfig();
    if (!latest.gatewayUrl) {
      if (!silent) {
        vscode.window.showWarningMessage("No organizer AI gateway URL is configured. Set aiEngineering.gatewayUrl or AI_GATEWAY_URL.");
      }
      return undefined;
    }
    if (!latest.gatewayToken) {
      if (!silent) {
        vscode.window.showWarningMessage("No AI gateway session token is available in AI_GATEWAY_SESSION_TOKEN.");
      }
      return undefined;
    }
    return new AiClient(createJsonRequester(latest.gatewayUrl, 20000), latest.gatewayToken);
  }

  async function refreshAllowance(): Promise<void> {
    const client = await getAiClient(true);
    if (!client) {
      aiProvider.setAllowance(undefined);
      return;
    }
    try {
      aiProvider.setAllowance(await client.allowance());
    } catch (error) {
      aiProvider.setMessage(`AI gateway unavailable: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

export function deactivate(): void {}

function currentConfig() {
  return resolveConfigValues(vscode.workspace.getConfiguration("aiEngineering"));
}
