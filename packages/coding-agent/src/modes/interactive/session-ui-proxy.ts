/**
 * SessionUiProxy — a stable, per-session ExtensionUIContext for multi-agent
 * foreground switching.
 *
 * Each live AgentSession is bound to its proxy exactly once (bindExtensions or
 * AgentSession.setExtensionUiContext). Extensions always hold this same context
 * object; switching the foreground agent only flips the proxy's internal
 * `active` flag:
 *
 *   - active  → calls forward to the real InteractiveMode UI context
 *   - dormant → calls are recorded in the proxy's footprint (widgets, statuses,
 *               footer, header, title, working state, terminal-input handlers,
 *               autocomplete providers, custom editor) so they can be replayed
 *               when the session becomes foreground again
 *
 * Because the bound context object never changes, no re-bind is needed on
 * switch and `session_start` is never re-emitted (re-binding on switch would
 * duplicate session_start and terminal-input listeners).
 */

import type { Component, OverlayHandle, OverlayOptions, TUI } from "@earendil-works/pi-tui";
import type {
	AutocompleteProviderFactory,
	EditorFactory,
	ExtensionAgentsApi,
	ExtensionUIContext,
	ExtensionUIDialogOptions,
	ExtensionWidgetOptions,
	TerminalInputHandler,
	WorkingIndicatorOptions,
} from "../../core/extensions/types.ts";
import type { ReadonlyFooterDataProvider } from "../../core/footer-data-provider.ts";
import type { KeybindingsManager } from "../../core/keybindings.ts";
import type { Theme } from "./theme/theme.ts";

type DisposableComponent = Component & { dispose?(): void };
type WidgetFactory = (tui: TUI, theme: Theme) => DisposableComponent;
type WidgetContent = string[] | WidgetFactory | undefined;
type FooterFactory =
	| ((tui: TUI, theme: Theme, footerData: ReadonlyFooterDataProvider) => DisposableComponent)
	| undefined;
type HeaderFactory = WidgetFactory | undefined;

/** Delegate used for interactive requests while the session is backgrounded. */
export type DormantUiDelegate = Partial<
	Pick<ExtensionUIContext, "select" | "confirm" | "input" | "editor" | "custom" | "notify">
>;

export interface SessionUiProxyOptions {
	/**
	 * Handles dialogs/notifications while dormant (e.g. a subagent runner's
	 * pending-UI context that surfaces needs_attention). When omitted, dormant
	 * dialogs auto-cancel and notifications are queued until foregrounded.
	 */
	dormantDelegate?: DormantUiDelegate;
}

const MAX_QUEUED_NOTIFICATIONS = 50;

export class SessionUiProxy implements ExtensionUIContext {
	readonly id: string;
	private readonly direct: ExtensionUIContext;
	private readonly dormantDelegate: DormantUiDelegate | undefined;

	private active = false;

	// Remembered footprint (survives backgrounding).
	private readonly widgets = new Map<
		string,
		{ content: WidgetContent; options: ExtensionWidgetOptions | undefined }
	>();
	private readonly statuses = new Map<string, string>();
	private footerFactory: FooterFactory;
	private headerFactory: HeaderFactory;
	private title: string | undefined;
	private workingMessage: string | undefined;
	private workingMessageSet = false;
	private workingVisible = true;
	private workingIndicator: WorkingIndicatorOptions | undefined;
	private workingIndicatorSet = false;
	private hiddenThinkingLabel: string | undefined;
	private hiddenThinkingLabelSet = false;
	private editorFactory: EditorFactory | undefined;
	private editorFactorySet = false;
	private readonly autocompleteProviders: AutocompleteProviderFactory[] = [];
	/** handler → live unsubscribe from the direct context (undefined while dormant) */
	private readonly inputHandlers = new Map<TerminalInputHandler, (() => void) | undefined>();
	private readonly queuedNotifications: Array<{ message: string; type: "info" | "warning" | "error" | undefined }> =
		[];

	constructor(id: string, direct: ExtensionUIContext, options: SessionUiProxyOptions = {}) {
		this.id = id;
		this.direct = direct;
		this.dormantDelegate = options.dormantDelegate;
	}

	get isActive(): boolean {
		return this.active;
	}

	// ---- lifecycle, driven by the host ----

	/** Foreground this session: replay the remembered footprint onto the terminal. */
	activate(): void {
		if (this.active) return;
		this.active = true;
		for (const [key, widget] of this.widgets) this.forwardWidget(key, widget.content, widget.options);
		for (const [key, text] of this.statuses) this.direct.setStatus(key, text);
		if (this.footerFactory) this.direct.setFooter(this.footerFactory);
		if (this.headerFactory) this.direct.setHeader(this.headerFactory);
		if (this.title !== undefined) this.direct.setTitle(this.title);
		if (this.workingMessageSet) this.direct.setWorkingMessage(this.workingMessage);
		if (!this.workingVisible) this.direct.setWorkingVisible(false);
		if (this.workingIndicatorSet) this.direct.setWorkingIndicator(this.workingIndicator);
		if (this.hiddenThinkingLabelSet) this.direct.setHiddenThinkingLabel(this.hiddenThinkingLabel);
		if (this.editorFactorySet) this.direct.setEditorComponent(this.editorFactory);
		for (const factory of this.autocompleteProviders) this.direct.addAutocompleteProvider(factory);
		for (const handler of this.inputHandlers.keys()) {
			this.inputHandlers.set(handler, this.direct.onTerminalInput(handler));
		}
		for (const notification of this.queuedNotifications.splice(0)) {
			this.direct.notify(notification.message, notification.type);
		}
	}

	/**
	 * Background this session: clear this session's terminal footprint but keep
	 * it remembered for the next activation.
	 *
	 * Note: autocomplete providers cannot be individually removed from the host;
	 * the host clears its wrapper list between deactivate/activate (see
	 * InteractiveMode.setForegroundAgent).
	 */
	deactivate(): void {
		if (!this.active) return;
		this.active = false;
		for (const [key, widget] of this.widgets) this.forwardWidget(key, undefined, widget.options);
		for (const key of this.statuses.keys()) this.direct.setStatus(key, undefined);
		if (this.footerFactory) this.direct.setFooter(undefined);
		if (this.headerFactory) this.direct.setHeader(undefined);
		if (this.workingMessageSet) this.direct.setWorkingMessage();
		if (!this.workingVisible) this.direct.setWorkingVisible(true);
		if (this.workingIndicatorSet) this.direct.setWorkingIndicator();
		if (this.hiddenThinkingLabelSet) this.direct.setHiddenThinkingLabel();
		if (this.editorFactorySet) this.direct.setEditorComponent(undefined);
		for (const [handler, unsubscribe] of this.inputHandlers) {
			unsubscribe?.();
			this.inputHandlers.set(handler, undefined);
		}
	}

	/**
	 * Single narrowing point for the widget overloads: the recorded union is
	 * passed back through the direct context, which accepts both shapes.
	 */
	private forwardWidget(key: string, content: WidgetContent, options: ExtensionWidgetOptions | undefined): void {
		(this.direct.setWidget as (key: string, content: WidgetContent, options?: ExtensionWidgetOptions) => void)(
			key,
			content,
			options,
		);
	}

	// ---- dialogs: foreground → real dialog; dormant → delegate or auto-cancel ----

	select(title: string, options: string[], opts?: ExtensionUIDialogOptions): Promise<string | undefined> {
		if (this.active) return this.direct.select(title, options, opts);
		if (this.dormantDelegate?.select) return this.dormantDelegate.select(title, options, opts);
		return Promise.resolve(undefined);
	}

	confirm(title: string, message: string, opts?: ExtensionUIDialogOptions): Promise<boolean> {
		if (this.active) return this.direct.confirm(title, message, opts);
		if (this.dormantDelegate?.confirm) return this.dormantDelegate.confirm(title, message, opts);
		return Promise.resolve(false);
	}

	input(title: string, placeholder?: string, opts?: ExtensionUIDialogOptions): Promise<string | undefined> {
		if (this.active) return this.direct.input(title, placeholder, opts);
		if (this.dormantDelegate?.input) return this.dormantDelegate.input(title, placeholder, opts);
		return Promise.resolve(undefined);
	}

	editor(title: string, prefill?: string): Promise<string | undefined> {
		if (this.active) return this.direct.editor(title, prefill);
		if (this.dormantDelegate?.editor) return this.dormantDelegate.editor(title, prefill);
		return Promise.resolve(undefined);
	}

	custom<T>(
		factory: (
			tui: TUI,
			theme: Theme,
			keybindings: KeybindingsManager,
			done: (result: T) => void,
		) => DisposableComponent | Promise<DisposableComponent>,
		options?: {
			overlay?: boolean;
			overlayOptions?: OverlayOptions | (() => OverlayOptions);
			onHandle?: (handle: OverlayHandle) => void;
		},
	): Promise<T> {
		if (this.active) return this.direct.custom(factory, options);
		if (this.dormantDelegate?.custom) return this.dormantDelegate.custom(factory, options);
		return Promise.resolve(undefined as T);
	}

	notify(message: string, type?: "info" | "warning" | "error"): void {
		if (this.active) {
			this.direct.notify(message, type);
			return;
		}
		if (this.dormantDelegate?.notify) {
			this.dormantDelegate.notify(message, type);
			return;
		}
		this.queuedNotifications.push({ message, type });
		if (this.queuedNotifications.length > MAX_QUEUED_NOTIFICATIONS) this.queuedNotifications.shift();
	}

	// ---- recorded, replayable terminal footprint ----

	onTerminalInput(handler: TerminalInputHandler): () => void {
		this.inputHandlers.set(handler, this.active ? this.direct.onTerminalInput(handler) : undefined);
		return () => {
			this.inputHandlers.get(handler)?.();
			this.inputHandlers.delete(handler);
		};
	}

	setStatus(key: string, text: string | undefined): void {
		if (text === undefined) this.statuses.delete(key);
		else this.statuses.set(key, text);
		if (this.active) this.direct.setStatus(key, text);
	}

	setWorkingMessage(message?: string): void {
		this.workingMessage = message;
		this.workingMessageSet = message !== undefined;
		if (this.active) this.direct.setWorkingMessage(message);
	}

	setWorkingVisible(visible: boolean): void {
		this.workingVisible = visible;
		if (this.active) this.direct.setWorkingVisible(visible);
	}

	setWorkingIndicator(options?: WorkingIndicatorOptions): void {
		this.workingIndicator = options;
		this.workingIndicatorSet = options !== undefined;
		if (this.active) this.direct.setWorkingIndicator(options);
	}

	setHiddenThinkingLabel(label?: string): void {
		this.hiddenThinkingLabel = label;
		this.hiddenThinkingLabelSet = label !== undefined;
		if (this.active) this.direct.setHiddenThinkingLabel(label);
	}

	setWidget(key: string, content: string[] | undefined, options?: ExtensionWidgetOptions): void;
	setWidget(key: string, content: WidgetFactory | undefined, options?: ExtensionWidgetOptions): void;
	setWidget(key: string, content: WidgetContent, options?: ExtensionWidgetOptions): void {
		if (content === undefined) this.widgets.delete(key);
		else this.widgets.set(key, { content, options });
		if (this.active) this.forwardWidget(key, content, options);
	}

	setFooter(factory: FooterFactory): void {
		this.footerFactory = factory;
		if (this.active) this.direct.setFooter(factory);
	}

	setHeader(factory: HeaderFactory): void {
		this.headerFactory = factory;
		if (this.active) this.direct.setHeader(factory);
	}

	setTitle(title: string): void {
		this.title = title;
		if (this.active) this.direct.setTitle(title);
	}

	addAutocompleteProvider(factory: AutocompleteProviderFactory): void {
		this.autocompleteProviders.push(factory);
		if (this.active) this.direct.addAutocompleteProvider(factory);
	}

	setEditorComponent(factory: EditorFactory | undefined): void {
		this.editorFactory = factory;
		this.editorFactorySet = factory !== undefined;
		if (this.active) this.direct.setEditorComponent(factory);
	}

	getEditorComponent(): EditorFactory | undefined {
		return this.active ? this.direct.getEditorComponent() : this.editorFactory;
	}

	// ---- shared-editor surface: only the foreground session may touch it ----

	pasteToEditor(text: string): void {
		if (this.active) this.direct.pasteToEditor(text);
	}

	setEditorText(text: string): void {
		if (this.active) this.direct.setEditorText(text);
	}

	getEditorText(): string {
		return this.active ? this.direct.getEditorText() : "";
	}

	// ---- global (terminal-wide) state: reads pass through, writes need foreground ----

	get theme(): Theme {
		return this.direct.theme;
	}

	getAllThemes(): { name: string; path: string | undefined }[] {
		return this.direct.getAllThemes();
	}

	getTheme(name: string): Theme | undefined {
		return this.direct.getTheme(name);
	}

	setTheme(themeOrName: string | Theme): { success: boolean; error?: string } {
		if (!this.active) return { success: false, error: "Background agents cannot change the theme" };
		return this.direct.setTheme(themeOrName);
	}

	getToolsExpanded(): boolean {
		return this.direct.getToolsExpanded();
	}

	setToolsExpanded(expanded: boolean): void {
		if (this.active) this.direct.setToolsExpanded(expanded);
	}

	/** Host-level agents API is session-independent; expose it on every proxy. */
	get agents(): ExtensionAgentsApi | undefined {
		return this.direct.agents;
	}
}
