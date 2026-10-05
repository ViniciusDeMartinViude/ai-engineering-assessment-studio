# AI Engineering Assessment VS Code Extension

Student-facing VS Code client for M9. The extension talks to:

- the local assessment agent at `aiEngineering.agentUrl`
- the organizer AI gateway at `aiEngineering.gatewayUrl` or `AI_GATEWAY_URL`

It does not call OpenAI directly and does not contain provider API keys, scoring logic, hidden tests, physical robot credentials, or organizer secrets.

## Develop

```bat
npm install
npm run compile
npm test
```

Start the local agent from the repository root before opening the extension development host:

```bat
python -m ai_assessment.local_agent --workspace candidate_workspaces\C014 --port 8787
```

## Package

```bat
npm run package
```

This uses `@vscode/vsce` to create a `.vsix`. Do not commit `node_modules`, `out`, or generated `.vsix` files.

## Views

- **Assessment** shows candidate, session, workspace, agent, gateway, and robot state reported by the backend.
- **AI Assistant** sends explicit selected context to the organizer gateway and displays responses.
- **Robot** forwards P1-P5, suction, health, positions, and backend-approved XYZ commands through the local agent.

AI code edits are deliberate. The extension can preview, insert, replace, or open a code block, but it never automatically saves, runs, submits, or sends AI-generated code to the robot.
