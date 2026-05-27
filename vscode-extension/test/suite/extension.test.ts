/**
 * Integration tests for the Surreal-Memory VS Code extension.
 *
 * Runs inside the VS Code Extension Development Host via @vscode/test-electron.
 */

import * as assert from "assert";
import * as vscode from "vscode";

suite("Extension", () => {
  test("extension should be present", () => {
    const ext = vscode.extensions.getExtension("surrealmemory.surrealmemory");
    assert.ok(ext, "Extension not found in registry");
  });

  test("all commands should be registered", async () => {
    const allCommands = await vscode.commands.getCommands(true);

    const expected = [
      "surrealmemory.encode",
      "surrealmemory.encodeInput",
      "surrealmemory.recall",
      "surrealmemory.openGraph",
      "surrealmemory.switchBrain",
      "surrealmemory.createBrain",
      "surrealmemory.refreshMemories",
      "surrealmemory.startServer",
      "surrealmemory.connectServer",
      "surrealmemory.recallFromTree",
    ];

    for (const cmd of expected) {
      assert.ok(
        allCommands.includes(cmd),
        `Command "${cmd}" not registered`,
      );
    }
  });

  test("configuration defaults should be set", () => {
    const config = vscode.workspace.getConfiguration("surrealmemory");

    assert.strictEqual(config.get("pythonPath"), "python");
    assert.strictEqual(config.get("autoStart"), false);
    assert.strictEqual(config.get("serverUrl"), "http://127.0.0.1:8000");
    assert.strictEqual(config.get("graphNodeLimit"), 1000);
    assert.strictEqual(config.get("codeLensEnabled"), true);

    const triggers = config.get<string[]>("commentTriggers");
    assert.ok(Array.isArray(triggers));
    assert.ok(triggers!.includes("remember:"));
    assert.ok(triggers!.includes("note:"));
    assert.ok(triggers!.includes("decision:"));
    assert.ok(triggers!.includes("todo:"));
  });
});
