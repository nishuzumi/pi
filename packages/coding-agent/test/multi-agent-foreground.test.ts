/**
 * End-to-end multi-agent foreground switching against a REAL InteractiveMode
 * rendered into a virtual terminal (@xterm/headless).
 *
 * Proves, at the pixel level:
 *  - editor input routes to the current foreground agent's session
 *  - foregrounding a background agent repaints the scrollback with ITS conversation
 *  - a mid-stream foreground attach renders the in-flight partial message
 *  - background agents keep streaming without painting the terminal
 *  - extension UI footprint (widget) follows the foreground agent
 *  - session_start fires exactly once per session across all switches
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { VirtualTerminal } from "../../tui/test/virtual-terminal.ts";
import type { AgentSession } from "../src/core/agent-session.ts";
import { type CreateAgentSessionRuntimeFactory, createAgentSessionRuntime } from "../src/core/agent-session-runtime.ts";
import { createAgentSessionFromServices, createAgentSessionServices } from "../src/core/agent-session-services.ts";
import type { ExtensionAgentsApi, ExtensionAPI } from "../src/core/extensions/index.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

// ---------------------------------------------------------------------------
// Mock provider: word-by-word streaming, no network. The reply text embeds the
// session tag + prompt so viewport assertions can tell agents apart.
// ---------------------------------------------------------------------------

const DELTA_MS = 20;

// NOTE: streamSimple registration is process-global in pi-ai (the last
// registerProvider("mock") wins), so a single shared stream serves BOTH
// sessions. The reply tag comes from each session's own system prompt.
function sharedMockStream(
	model: { api: string; provider: string; id: string },
	context: unknown,
	options?: { signal?: AbortSignal },
) {
	const stream = createAssistantMessageEventStream();
	(async () => {
		const systemPrompt = String((context as { systemPrompt?: string }).systemPrompt ?? "");
		const tag = /TAG:(\w+)/.exec(systemPrompt)?.[1] ?? "UNKNOWN";
		{
			const messages = (context as { messages?: Array<{ role: string; content: unknown }> }).messages ?? [];
			const lastUser = [...messages].reverse().find((message) => message.role === "user");
			const userText =
				typeof lastUser?.content === "string"
					? lastUser.content
					: ((lastUser?.content as Array<{ type: string; text?: string }> | undefined)
							?.filter((part) => part.type === "text")
							.map((part) => part.text)
							.join(" ") ?? "");
			const words = `reply-from-${tag} to [${userText}] alpha beta gamma delta epsilon zeta eta theta`.split(" ");
			await streamWords(stream, model, words, options);
		}
	})();
	return stream;
}

async function streamWords(
	stream: ReturnType<typeof createAssistantMessageEventStream>,
	model: { api: string; provider: string; id: string },
	words: string[],
	options?: { signal?: AbortSignal },
) {
	{
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 1,
				output: words.length,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 1 + words.length,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		} as AssistantMessage;
		try {
			stream.push({ type: "start", partial: output });
			output.content.push({ type: "text", text: "" });
			stream.push({ type: "text_start", contentIndex: 0, partial: output });
			for (const word of words) {
				if (options?.signal?.aborted) throw new Error("aborted");
				await new Promise((resolve) => setTimeout(resolve, DELTA_MS));
				const block = output.content[0] as { type: "text"; text: string };
				block.text += block.text.length === 0 ? word : ` ${word}`;
				stream.push({ type: "text_delta", contentIndex: 0, delta: word, partial: output });
			}
			stream.push({
				type: "text_end",
				contentIndex: 0,
				content: (output.content[0] as { type: "text"; text: string }).text,
				partial: output,
			});
			stream.push({ type: "done", reason: "stop", message: output });
			stream.end();
		} catch (error) {
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			(output as { errorMessage?: string }).errorMessage = error instanceof Error ? error.message : String(error);
			stream.push({ type: "error", reason: output.stopReason as never, error: output as never });
			stream.end();
		}
	}
}

interface ProbeCounters {
	starts: number;
	shutdowns: number;
	agents?: ExtensionAgentsApi;
}

function makeExtensions(tag: string, counters: ProbeCounters) {
	return [
		{
			name: `probe-${tag}`,
			factory: (pi: ExtensionAPI) => {
				pi.registerProvider("mock", {
					baseUrl: "http://localhost:0",
					apiKey: "mock-key",
					api: "openai-completions",
					streamSimple: sharedMockStream as never,
					models: [
						{
							id: "mock-1",
							name: "Mock Model",
							reasoning: false,
							input: ["text"],
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
							contextWindow: 100000,
							maxTokens: 4096,
						},
					],
				} as never);
				pi.on("session_start", async (_event, ctx) => {
					counters.starts++;
					counters.agents = ctx.ui.agents;
					ctx.ui.setWidget(`${tag}-widget`, [`WIDGET-OF-${tag}`]);
				});
				pi.on("session_shutdown", async () => {
					counters.shutdowns++;
				});
			},
		},
	];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(
	terminal: VirtualTerminal,
	predicate: () => boolean,
	what: string,
	timeoutMs = 5000,
): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		await terminal.waitForRender();
		if (predicate()) return;
		await sleep(25);
	}
	throw new Error(`Timed out waiting for: ${what}
--- viewport ---
${terminal.getViewport().join("\n")}
--- end viewport ---`);
}

function viewportText(terminal: VirtualTerminal): string {
	return terminal.getViewport().join("\n");
}

function scrollText(terminal: VirtualTerminal): string {
	return terminal.getScrollBuffer().join("\n");
}

async function pinMockModel(session: AgentSession): Promise<void> {
	const model = session.modelRegistry.find("mock", "mock-1");
	if (!model) throw new Error("mock model not registered");
	await session.setModel(model);
}

function lastAssistantText(session: AgentSession): string {
	const assistant = [...session.messages].reverse().find((message) => message.role === "assistant") as
		| { content?: Array<{ type: string; text?: string }> }
		| undefined;
	return assistant?.content?.find((part) => part.type === "text")?.text ?? "";
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

describe("multi-agent foreground switching (virtual terminal e2e)", () => {
	let tempDir: string;
	let terminal: VirtualTerminal;
	let mode: InteractiveMode;
	let mainSession: AgentSession;
	let childSession: AgentSession;
	let detachChild: (() => void) | undefined;
	const mainCounters: ProbeCounters = { starts: 0, shutdowns: 0 };
	const childCounters: ProbeCounters = { starts: 0, shutdowns: 0 };

	beforeAll(async () => {
		process.env.PI_OFFLINE = "1";
		initTheme("dark");
		tempDir = mkdtempSync(join(tmpdir(), "pi-multi-agent-test-"));

		// --- main session runtime (hermetic: temp agentDir, in-memory settings/session) ---
		const agentDir = join(tempDir, "agent");
		const settingsManager = SettingsManager.inMemory({ quietStartup: true });
		const createRuntime: CreateAgentSessionRuntimeFactory = async ({ cwd, sessionManager, sessionStartEvent }) => {
			const services = await createAgentSessionServices({
				cwd,
				agentDir,
				settingsManager,
				resourceLoaderOptions: {
					noExtensions: true,
					noSkills: true,
					noPromptTemplates: true,
					noContextFiles: true,
					systemPrompt: "You are the MAIN agent. TAG:MAIN",
					extensionFactories: makeExtensions("MAIN", mainCounters),
				},
			});
			return {
				...(await createAgentSessionFromServices({ services, sessionManager, sessionStartEvent })),
				services,
				diagnostics: services.diagnostics,
			};
		};
		const runtime = await createAgentSessionRuntime(createRuntime, {
			cwd: tempDir,
			agentDir,
			sessionManager: SessionManager.inMemory(tempDir),
		});
		mainSession = runtime.session;

		// --- interactive mode on a virtual terminal ---
		terminal = new VirtualTerminal(120, 36);
		mode = new InteractiveMode(runtime, { terminal });
		void mode.run().catch(() => {
			// run() pends forever on user input; errors surface via assertions.
		});
		await waitFor(terminal, () => mainCounters.starts === 1, "main session_start");
		await pinMockModel(mainSession);

		// --- child session (separate live AgentSession, own extensions) ---
		const childSettings = SettingsManager.inMemory({ quietStartup: true });
		const childLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager: childSettings,
			noExtensions: true,
			noSkills: true,
			noPromptTemplates: true,
			noContextFiles: true,
			systemPrompt: "You are the CHILD agent. TAG:CHILD",
			extensionFactories: makeExtensions("CHILD", childCounters),
		});
		await childLoader.reload();
		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			resourceLoader: childLoader,
			sessionManager: SessionManager.inMemory(tempDir),
			settingsManager: childSettings,
		});
		childSession = session;

		// Adopt BEFORE bind so the child's one and only bind uses the host proxy
		// (footprint recorded from the very first session_start).
		const agents = mainCounters.agents;
		if (!agents) throw new Error("agents api not exposed to main extension");
		detachChild = agents.adopt("child", childSession, { label: "child" });
		await childSession.bindExtensions({ mode: "tui" });
		await pinMockModel(childSession);
	}, 30000);

	afterAll(async () => {
		detachChild?.();
		childSession?.dispose();
		(mode as unknown as { stop(): void }).stop();
		rmSync(tempDir, { recursive: true, force: true });
	});

	it("routes editor input to main and renders its reply", async () => {
		terminal.sendInput("hello-main");
		terminal.sendInput("\r");
		await waitFor(
			terminal,
			() => scrollText(terminal).includes("reply-from-MAIN to [hello-main]"),
			"main reply visible",
		);
		await mainSession.waitForIdle();
		expect(lastAssistantText(mainSession)).toContain("reply-from-MAIN to [hello-main]");
	});

	it("shows the main extension widget while main is foreground", async () => {
		await waitFor(terminal, () => viewportText(terminal).includes("WIDGET-OF-MAIN"), "main widget visible");
		expect(viewportText(terminal)).not.toContain("WIDGET-OF-CHILD");
	});

	it("background child streams without painting the terminal", async () => {
		const childRun = childSession.prompt("child-task");
		await sleep(120);
		expect(childSession.isStreaming).toBe(true);
		expect(scrollText(terminal)).not.toContain("reply-from-CHILD");
		await childRun;
		await childSession.waitForIdle();
		expect(lastAssistantText(childSession)).toContain("reply-from-CHILD to [child-task]");
		expect(scrollText(terminal)).not.toContain("reply-from-CHILD");
	});

	it("foregrounding the child renders ITS conversation into the flow", async () => {
		const agents = mainCounters.agents;
		if (!agents) throw new Error("agents api missing");
		await agents.setForeground("child");
		await waitFor(
			terminal,
			() => viewportText(terminal).includes("reply-from-CHILD to [child-task]"),
			"child conversation visible",
		);
		// Flow render: the child's document is APPENDED below the old foreground's
		// content (which scrolls into native scrollback) — assert ordering, and
		// that the live UI footprint now belongs to the child.
		const scroll = scrollText(terminal);
		expect(scroll.lastIndexOf("reply-from-CHILD to [child-task]")).toBeGreaterThan(
			scroll.lastIndexOf("reply-from-MAIN"),
		);
		expect(scroll.lastIndexOf("WIDGET-OF-CHILD")).toBeGreaterThan(scroll.lastIndexOf("WIDGET-OF-MAIN"));
	});

	it("routes editor input to the child while it is foreground", async () => {
		const mainMessagesBefore = mainSession.messages.length;
		terminal.sendInput("second-child-task");
		terminal.sendInput("\r");
		await waitFor(
			terminal,
			() => scrollText(terminal).includes("reply-from-CHILD to [second-child-task]"),
			"child second reply visible",
		);
		await childSession.waitForIdle();
		expect(lastAssistantText(childSession)).toContain("reply-from-CHILD to [second-child-task]");
		expect(mainSession.messages.length).toBe(mainMessagesBefore);
	});

	it("attaches mid-stream: partial output renders after a foreground switch", async () => {
		const agents = mainCounters.agents;
		if (!agents) throw new Error("agents api missing");
		// Back to main, then start a slow child turn in the background.
		await agents.setForeground("main");
		await waitFor(terminal, () => viewportText(terminal).includes("WIDGET-OF-MAIN"), "main foreground again");
		const childRun = childSession.prompt("midstream-task");
		await sleep(80); // child is now mid-stream
		expect(childSession.isStreaming).toBe(true);
		await agents.setForeground("child");
		// The seeded streaming component + live deltas must render the reply.
		await waitFor(
			terminal,
			() => scrollText(terminal).includes("reply-from-CHILD to [midstream-task]"),
			"mid-stream child reply visible",
		);
		await childRun;
		await childSession.waitForIdle();
		expect(lastAssistantText(childSession)).toContain("reply-from-CHILD to [midstream-task]");
	});

	it("switches back to main with its conversation and widget intact", async () => {
		const agents = mainCounters.agents;
		if (!agents) throw new Error("agents api missing");
		await agents.setForeground("main");
		await waitFor(terminal, () => viewportText(terminal).includes("WIDGET-OF-MAIN"), "main widget replayed");
		// Main's full conversation was re-appended into the flow below the child's,
		// and the live widget footprint is main's again.
		const scroll = scrollText(terminal);
		expect(scroll.lastIndexOf("reply-from-MAIN to [hello-main]")).toBeGreaterThan(
			scroll.lastIndexOf("reply-from-CHILD"),
		);
		expect(scroll.lastIndexOf("WIDGET-OF-MAIN")).toBeGreaterThan(scroll.lastIndexOf("WIDGET-OF-CHILD"));
	});

	it("never re-emitted session_start or session_shutdown during switching", () => {
		expect(mainCounters.starts).toBe(1);
		expect(childCounters.starts).toBe(1);
		expect(mainCounters.shutdowns).toBe(0);
		expect(childCounters.shutdowns).toBe(0);
	});

	it("lists agents with correct foreground marker", () => {
		const agents = mainCounters.agents;
		if (!agents) throw new Error("agents api missing");
		expect(agents.list()).toEqual([
			{ id: "main", label: "main", isForeground: true },
			{ id: "child", label: "child", isForeground: false },
		]);
	});
});
