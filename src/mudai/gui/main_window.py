"""Main application window: wires MUD client, agent, GUI, and decision loop."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..llm.agent import Agent, Decision
from ..llm.backend import LlamaBackend
from ..llm.chat import ChatSession
from ..llm.memory import MemoryStore
from ..logging.trace_logger import TraceLogger
from ..mud.ansi import strip_ansi
from ..mud.telnet_client import MudClient
from .chat_panel import ChatPanel
from .memory_panel import MemoryPanel
from .settings_dialog import SettingsDialog
from .widgets import mono_font, MudOutputView, ReasoningView


class MainWindow(QMainWindow):
    """Primary window. Lives on the Qt thread; uses asyncio (qasync) for IO/LLM."""

    # Signals so worker callbacks can post to the GUI thread safely.
    sig_mud_text = pyqtSignal(str)
    sig_status = pyqtSignal(str)
    sig_reason_delta = pyqtSignal(str)
    sig_decision_done = pyqtSignal(object)  # Decision
    sig_decision_error = pyqtSignal(str)
    sig_chat_delta = pyqtSignal(str)
    sig_chat_done = pyqtSignal(str)            # full assistant text
    sig_chat_error = pyqtSignal(str)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("MUD.ai")
        self.resize(1500, 950)

        # State
        self.backend = LlamaBackend(cfg.model_path(), cfg.llm)
        from ..config import MEMORY_PATH
        self.memory = MemoryStore.load(MEMORY_PATH)
        self.agent = Agent(self.backend, cfg.agent, cfg.llm, memory=self.memory)
        self.chat = ChatSession(self.backend, self.agent, cfg.agent, cfg.llm)
        self.mud: MudClient | None = None
        self.trace_logger = TraceLogger()
        self._decision_inflight = False
        self._chat_inflight = False
        self._last_send_ms = 0.0
        self._pending_decision: Decision | None = None
        self._pending_outcome_buf: list[str] = []
        self._pending_last_log: dict[str, Any] | None = None
        # Manual-input command history (newest last).
        self._cmd_history: list[str] = []
        self._cmd_history_idx: int = 0

        # Idle-debounce timer for triggering decisions.
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._on_idle_elapsed)

        self._build_ui()
        self._build_menus()

        # Signal wiring (always queued -> safe across threads).
        self.sig_mud_text.connect(self._on_mud_text)
        self.sig_status.connect(self._on_status)
        self.sig_reason_delta.connect(self.reasoning_view.append_delta)
        self.sig_decision_done.connect(self._on_decision_done)
        self.sig_decision_error.connect(self._on_decision_error)
        self.sig_chat_delta.connect(self.chat_panel.append_assistant_delta)
        self.sig_chat_done.connect(self._on_chat_done)
        self.sig_chat_error.connect(self._on_chat_error)

        self._refresh_autonomy_label()
        self._update_status_bar()

        # Optional auto-start: schedule after event loop is running.
        QTimer.singleShot(250, self._maybe_autostart)

    def _maybe_autostart(self) -> None:
        if self.cfg.agent.auto_load_model_on_start and self.cfg.model_path().exists():
            self._on_load_model_clicked()
        if self.cfg.mud.auto_connect_on_start:
            self._on_connect_clicked()

    # ----- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        self.mud_view = MudOutputView()
        self.reasoning_view = ReasoningView()

        self.steering_edit = QPlainTextEdit()
        self.steering_edit.setFont(mono_font())
        self.steering_edit.setPlainText(self.cfg.agent.steering_notes)
        self.steering_edit.setPlaceholderText(
            "Steering notes - injected into every decision's system prompt. "
            "Edit any time."
        )
        self.steering_edit.textChanged.connect(self._on_steering_changed)

        steering_label = QLabel("Steering notes (injected into system prompt):")
        steering_container = QWidget()
        sc_lay = QVBoxLayout(steering_container)
        sc_lay.setContentsMargins(0, 0, 0, 0)
        sc_lay.addWidget(steering_label)
        sc_lay.addWidget(self.steering_edit)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self.reasoning_view)
        right_split.addWidget(steering_container)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)

        self.chat_panel = ChatPanel()
        self.chat_panel.message_submitted.connect(self._on_chat_user_submit)
        self.chat_panel.clear_requested.connect(self._on_chat_clear)

        self.memory_panel = MemoryPanel(self.memory)

        # Reasoning/steering is always visible on top; chat & memory live in
        # a tabbed pane below it so the agent's gameloop is never hidden.
        right_tabs = QTabWidget()
        self._chat_tab_index = right_tabs.addTab(self.chat_panel, "Chat with agent")
        right_tabs.addTab(self.memory_panel, "Memory")
        self.right_tabs = right_tabs

        right_pane_split = QSplitter(Qt.Orientation.Vertical)
        right_pane_split.addWidget(right_split)
        right_pane_split.addWidget(right_tabs)
        right_pane_split.setStretchFactor(0, 3)
        right_pane_split.setStretchFactor(1, 2)

        # State for the orange attention-flash on the chat tab.
        self._chat_flash_timer = QTimer(self)
        self._chat_flash_timer.setInterval(500)
        self._chat_flash_on = False
        self._chat_flash_timer.timeout.connect(self._tick_chat_flash)
        right_tabs.currentChanged.connect(self._on_right_tab_changed)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(self.mud_view)
        main_split.addWidget(right_pane_split)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 2)

        # Bottom action bar: proposed command + buttons + manual input.
        self.proposed_edit = QLineEdit()
        self.proposed_edit.setPlaceholderText("Proposed command from LLM will appear here. You can edit before sending.")
        self.proposed_edit.setFont(mono_font())
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send_proposed)
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self._on_reject_proposed)
        self.think_btn = QPushButton("Think now")
        self.think_btn.clicked.connect(self._trigger_decision_now)
        self.note_btn = QPushButton("Inject note")
        self.note_btn.setToolTip(
            "Add a one-shot operator note into the agent's context "
            "(not sent to the MUD)."
        )
        self.note_btn.clicked.connect(self._on_inject_note)
        self.auto_btn = QPushButton("AUTO: OFF")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(self.cfg.agent.auto_send)
        self.auto_btn.toggled.connect(self._on_toggle_auto)

        proposal_row = QHBoxLayout()
        proposal_row.addWidget(QLabel("Proposed:"))
        proposal_row.addWidget(self.proposed_edit, 1)
        proposal_row.addWidget(self.send_btn)
        proposal_row.addWidget(self.reject_btn)
        proposal_row.addWidget(self.think_btn)
        proposal_row.addWidget(self.note_btn)
        proposal_row.addWidget(self.auto_btn)

        self.manual_edit = QLineEdit()
        self.manual_edit.setPlaceholderText("Manual command (sent directly to MUD on Enter)")
        self.manual_edit.setFont(mono_font())
        self.manual_edit.returnPressed.connect(self._on_manual_send)
        # Up/Down arrows scroll command history.
        self.manual_edit.installEventFilter(self)

        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("You:    "))
        manual_row.addWidget(self.manual_edit, 1)

        bottom = QWidget()
        b_lay = QVBoxLayout(bottom)
        b_lay.addLayout(proposal_row)
        b_lay.addLayout(manual_row)

        central = QWidget()
        c_lay = QVBoxLayout(central)
        c_lay.addWidget(main_split, 1)
        c_lay.addWidget(bottom)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar(self))

    def _build_menus(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_connect = QAction("Connect", self)
        self.act_connect.triggered.connect(self._on_connect_clicked)
        tb.addAction(self.act_connect)

        self.act_disconnect = QAction("Disconnect", self)
        self.act_disconnect.triggered.connect(self._on_disconnect_clicked)
        self.act_disconnect.setEnabled(False)
        tb.addAction(self.act_disconnect)

        tb.addSeparator()

        self.act_llm_toggle = QAction("LLM: OFF", self)
        self.act_llm_toggle.setCheckable(True)
        self.act_llm_toggle.setChecked(False)
        self.act_llm_toggle.toggled.connect(self._on_llm_toggle)
        tb.addAction(self.act_llm_toggle)

        tb.addSeparator()

        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(self._on_settings_clicked)
        tb.addAction(act_settings)

        act_clear = QAction("Clear views", self)
        act_clear.triggered.connect(self._on_clear_clicked)
        tb.addAction(act_clear)

    # ----- status / labels ---------------------------------------------------
    def _refresh_autonomy_label(self) -> None:
        on = self.auto_btn.isChecked()
        self.auto_btn.setText("AUTO: ON  [STOP]" if on else "AUTO: OFF")
        self.auto_btn.setStyleSheet(
            "QPushButton { background-color: #802020; color: white; font-weight: bold; }"
            if on else ""
        )

    def _update_status_bar(self) -> None:
        sb = self.statusBar()
        if sb is None:
            return
        conn = (
            f"MUD: {self.cfg.mud.host}:{self.cfg.mud.port} "
            f"({'connected' if (self.mud and self.mud.connected) else 'disconnected'})"
        )
        model = f"Model: {self.cfg.llm.model_file} ({'loaded' if self.backend.loaded else 'not loaded'})"
        ctx = f"n_ctx={self.cfg.llm.n_ctx} | budget={self.cfg.llm.transcript_token_budget} tok"
        sb.showMessage(f"{conn}   |   {model}   |   {ctx}")

    # ----- mud client callbacks (called from telnet reader task) -------------
    def _mud_on_text(self, text: str) -> None:
        self.sig_mud_text.emit(text)

    def _mud_on_status(self, status: str) -> None:
        self.sig_status.emit(status)

    def _on_mud_text(self, text: str) -> None:
        self.mud_view.append_mud(text)
        plain = strip_ansi(text)
        self.agent.add_mud(plain)
        if self._pending_decision is not None:
            self._pending_outcome_buf.append(plain)
        # Debounce: restart idle timer.
        self._idle_timer.start(max(50, self.cfg.mud.decision_idle_ms))

    def _on_status(self, status: str) -> None:
        sb = self.statusBar()
        if sb is not None:
            sb.showMessage(status, 4000)
        self.reasoning_view.append_note(f"[mud] {status}", color="#888")
        self._update_status_bar()
        # Toggle connect/disconnect actions if connection state changed.
        is_conn = bool(self.mud and self.mud.connected)
        self.act_connect.setEnabled(not is_conn)
        self.act_disconnect.setEnabled(is_conn)

    # ----- steering ----------------------------------------------------------
    def _on_steering_changed(self) -> None:
        self.cfg.agent.steering_notes = self.steering_edit.toPlainText()
        # Debounced save - cheap enough to save immediately.
        self.cfg.save()

    # ----- toolbar actions ---------------------------------------------------
    def _on_connect_clicked(self) -> None:
        if self.mud is not None and self.mud.connected:
            return
        self.mud = MudClient(
            host=self.cfg.mud.host,
            port=self.cfg.mud.port,
            on_text=self._mud_on_text,
            on_status=self._mud_on_status,
            encoding=self.cfg.mud.encoding,
        )
        asyncio.ensure_future(self._connect_task())

    async def _connect_task(self) -> None:
        assert self.mud is not None
        try:
            await self.mud.connect()
        except Exception as e:
            QMessageBox.critical(self, "Connect failed", str(e))

    def _on_disconnect_clicked(self) -> None:
        if self.mud is None:
            return
        asyncio.ensure_future(self.mud.close())

    def _on_load_model_clicked(self) -> None:
        self.backend = LlamaBackend(self.cfg.model_path(), self.cfg.llm)
        self.agent.backend = self.backend
        self.chat.backend = self.backend
        self.reasoning_view.append_note("[llm] loading model ...", color="#aaa")
        self._refresh_llm_toggle(loading=True)
        asyncio.ensure_future(self._load_model_task())

    async def _load_model_task(self) -> None:
        try:
            await asyncio.to_thread(self.backend.load)
        except Exception as e:
            self.reasoning_view.append_note(f"[llm] load failed: {e}", color="#f88")
            QMessageBox.critical(self, "Model load failed", str(e))
            self._refresh_llm_toggle()
            return
        self.reasoning_view.append_note("[llm] model loaded", color="#8f8")
        self._refresh_llm_toggle()
        self._update_status_bar()

    def _on_unload_model_clicked(self) -> None:
        self.backend.unload()
        self.reasoning_view.append_note("[llm] model unloaded", color="#aaa")
        self._refresh_llm_toggle()
        self._update_status_bar()

    def _on_llm_toggle(self, checked: bool) -> None:
        # Reflect actual state in label; perform load/unload.
        if checked:
            if not self.backend.loaded:
                self._on_load_model_clicked()
            else:
                self._refresh_llm_toggle()
        else:
            if self.backend.loaded:
                self._on_unload_model_clicked()
            else:
                self._refresh_llm_toggle()

    def _refresh_llm_toggle(self, loading: bool = False) -> None:
        btn = self.act_llm_toggle
        btn.blockSignals(True)
        if loading:
            btn.setChecked(True)
            btn.setEnabled(False)
            btn.setText("LLM: loading...")
        else:
            on = self.backend.loaded
            btn.setChecked(on)
            btn.setEnabled(True)
            btn.setText("LLM: ON" if on else "LLM: OFF")
        btn.blockSignals(False)

    def _on_settings_clicked(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            old_file = self.cfg.llm.model_file
            dlg.apply_to(self.cfg)
            self.cfg.save()
            # If model file or context window changed, drop loaded model.
            if old_file != self.cfg.llm.model_file:
                self.backend.unload()
                self.backend = LlamaBackend(self.cfg.model_path(), self.cfg.llm)
                self.agent.backend = self.backend
                self.chat.backend = self.backend
            else:
                # Update cfg references on existing backend / agent / chat.
                self.backend.cfg = self.cfg.llm
                self.agent.llm_cfg = self.cfg.llm
                self.agent.agent_cfg = self.cfg.agent
                self.chat.llm_cfg = self.cfg.llm
                self.chat.agent_cfg = self.cfg.agent
            self._update_status_bar()

    def _on_clear_clicked(self) -> None:
        self.mud_view.clear()
        self.reasoning_view.clear()

    # ----- autonomy ----------------------------------------------------------
    def _on_toggle_auto(self, checked: bool) -> None:
        self.cfg.agent.auto_send = checked
        self.cfg.save()
        self._refresh_autonomy_label()

    # ----- manual send -------------------------------------------------------
    def _on_manual_send(self) -> None:
        text = self.manual_edit.text()
        if not text:
            return
        self.manual_edit.clear()
        # Add to history (dedupe consecutive).
        if not self._cmd_history or self._cmd_history[-1] != text:
            self._cmd_history.append(text)
            if len(self._cmd_history) > 500:
                self._cmd_history = self._cmd_history[-500:]
        self._cmd_history_idx = len(self._cmd_history)
        self._send_command(text, source="you")

    def eventFilter(self, obj: Any, ev: Any) -> bool:  # type: ignore[override]
        if obj is self.manual_edit and isinstance(ev, QKeyEvent) and ev.type() == QKeyEvent.Type.KeyPress:
            if ev.key() == Qt.Key.Key_Up:
                if self._cmd_history and self._cmd_history_idx > 0:
                    self._cmd_history_idx -= 1
                    self.manual_edit.setText(self._cmd_history[self._cmd_history_idx])
                return True
            if ev.key() == Qt.Key.Key_Down:
                if self._cmd_history_idx < len(self._cmd_history) - 1:
                    self._cmd_history_idx += 1
                    self.manual_edit.setText(self._cmd_history[self._cmd_history_idx])
                else:
                    self._cmd_history_idx = len(self._cmd_history)
                    self.manual_edit.clear()
                return True
        return super().eventFilter(obj, ev)

    def _on_inject_note(self) -> None:
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Inject operator note",
            "This note is added to the agent's transcript as [OPERATOR] context.\n"
            "It is NOT sent to the MUD. The next decision will see it.",
            "",
        )
        if not ok:
            return
        text = text.strip()
        if not text:
            return
        self.agent.add_operator_note(text)
        self.reasoning_view.append_note(f"[operator note] {text}", color="#ffcc66")

    def _send_command(self, text: str, source: str) -> None:
        if self.mud is None or not self.mud.connected:
            self.reasoning_view.append_note(f"[skip] not connected: {text!r}", "#f88")
            return
        self.mud_view.append_local(text, color="#ffcc66" if source == "you" else "#66ddff")
        self._last_send_ms = time.monotonic() * 1000.0
        if source == "agent":
            self.agent.add_agent(text)
        else:
            self.agent.add_user(text)
        asyncio.ensure_future(self.mud.send(text))

    # ----- decision loop -----------------------------------------------------
    def _on_idle_elapsed(self) -> None:
        if self._decision_inflight:
            return
        if not self.backend.loaded:
            return
        if self.mud is None or not self.mud.connected:
            return
        self._trigger_decision_now()

    def _trigger_decision_now(self) -> None:
        if self._decision_inflight:
            return
        if not self.backend.loaded:
            self.reasoning_view.append_note(
                "[llm] no model loaded - click 'Load model' first", "#f88"
            )
            return
        self._decision_inflight = True
        self.reasoning_view.start_new("decision")
        # Snapshot transcript for the trace.
        transcript_snapshot = [
            {"role": e.role, "text": e.text, "ts": e.ts}
            for e in list(self.agent.transcript)[-200:]
        ]
        self._pending_last_log = {
            "system": self.agent._system_message(),
            "steering": self.cfg.agent.steering_notes,
            "transcript": transcript_snapshot,
        }
        asyncio.ensure_future(self._run_decision())

    async def _run_decision(self) -> None:
        loop = asyncio.get_running_loop()

        def emit_delta(d: str) -> None:
            # Called from worker thread; bounce to GUI via signal.
            loop.call_soon_threadsafe(self.sig_reason_delta.emit, d)

        try:
            decision: Decision = await asyncio.to_thread(
                self.agent.decide_stream, emit_delta
            )
        except Exception as e:
            self.sig_decision_error.emit(str(e))
            return
        self.sig_decision_done.emit(decision)

    def _on_decision_done(self, decision: object) -> None:
        assert isinstance(decision, Decision)
        self._decision_inflight = False
        self._pending_decision = decision
        self._pending_outcome_buf = []
        self.proposed_edit.setText(decision.command)
        self.reasoning_view.append_note(
            f"[proposed] {decision.command!r}", color="#ffcc66"
        )
        # Auto-capture any REMEMBER: lines the model emitted (from think,
        # reasoning, or anywhere in the raw response) into permanent memory.
        captured = self.memory.capture_from_text(
            f"{decision.thinking}\n{decision.raw}", source="agent_decision"
        )
        for e in captured:
            self.reasoning_view.append_note(
                f"[memory+] {e.text}", color="#88f088"
            )
        # If the model expressed uncertainty via `QUESTION: ...` in either the
        # think block or the visible reasoning, surface it in the chat panel
        # so the operator can answer.
        self._maybe_surface_question(decision)
        # Auto-send?
        if self.cfg.agent.auto_send:
            now_ms = time.monotonic() * 1000.0
            if now_ms - self._last_send_ms >= self.cfg.agent.min_command_interval_ms:
                self._on_send_proposed()
            else:
                # Delay until min-interval elapses.
                wait = int(self.cfg.agent.min_command_interval_ms - (now_ms - self._last_send_ms))
                QTimer.singleShot(max(50, wait), self._on_send_proposed)

    def _on_decision_error(self, msg: str) -> None:
        self._decision_inflight = False
        self.reasoning_view.append_note(f"[llm error] {msg}", color="#f88")

    # ----- chat (side conversation) -----------------------------------------
    def _on_chat_user_submit(self, text: str) -> None:
        if not text.strip():
            return
        # Always allow chat even when no model loaded? Better: require backend.
        if not self.backend.loaded:
            self.chat_panel.append_system_note(
                "no model loaded - click 'Load model' first", color="#f88"
            )
            return
        if self._chat_inflight:
            self.chat_panel.append_system_note(
                "previous reply still streaming - try again in a moment",
                color="#f88",
            )
            return
        self.chat.add_user(text)
        # Crucial: also inject into the game-loop transcript as an [OPERATOR]
        # note so the very next decision sees what the operator just said.
        # This is what makes the side conversation actually STEER gameplay
        # within the current session (persistent learning happens via the LoRA
        # training pipeline over logs/traces/*.jsonl).
        self.agent.add_operator_note(f"(via chat) {text}")
        # Operator may also use the REMEMBER: <fact> shorthand directly.
        captured = self.memory.capture_from_text(text, source="operator")
        for e in captured:
            self.chat_panel.append_system_note(
                f"saved to memory: {e.text}", color="#88f088"
            )
        self.chat_panel.append_user(text)
        self.chat_panel.begin_assistant()
        self._chat_inflight = True
        asyncio.ensure_future(self._run_chat())

    def _on_chat_clear(self) -> None:
        self.chat.clear()
        self.chat_panel.clear()

    async def _run_chat(self) -> None:
        loop = asyncio.get_running_loop()

        def emit_delta(d: str) -> None:
            loop.call_soon_threadsafe(self.sig_chat_delta.emit, d)

        try:
            full = await asyncio.to_thread(self.chat.stream_reply, emit_delta)
        except Exception as e:
            self.sig_chat_error.emit(str(e))
            return
        self.sig_chat_done.emit(full)

    def _on_chat_done(self, full: str) -> None:
        self._chat_inflight = False
        self.chat_panel.end_assistant()
        # Auto-capture any REMEMBER: lines the model wrote in chat.
        captured = self.memory.capture_from_text(full, source="agent_chat")
        for e in captured:
            self.chat_panel.append_system_note(
                f"saved to memory: {e.text}", color="#88f088"
            )
        # Persist for fine-tuning. ``stream_reply`` already appended the
        # assistant turn to chat history; grab the prior user turn as the pair.
        history = list(self.chat.history)
        user_text = ""
        for turn in reversed(history[:-1]):
            if turn.role == "user":
                user_text = turn.text
                break
        transcript_snapshot = [
            {"role": e.role, "text": e.text, "ts": e.ts}
            for e in list(self.agent.transcript)[-200:]
        ]
        self.trace_logger.log({
            "type": "chat",
            "system": self.chat._system_with_context(),
            "steering": self.cfg.agent.steering_notes,
            "transcript": transcript_snapshot,
            "user": user_text,
            "assistant": full,
        })

    def _on_chat_error(self, msg: str) -> None:
        self._chat_inflight = False
        self.chat_panel.append_system_note(f"chat error: {msg}", color="#f88")
        self.chat_panel.end_assistant()

    def _maybe_surface_question(self, decision: Decision) -> None:
        """If the decision reasoning contains 'QUESTION: ...', pop it into chat.

        Does NOT steal focus from whatever tab the user is on; instead, flashes
        the chat tab orange until the user opens it.
        """
        haystack = f"{decision.thinking}\n{decision.reasoning}"
        question = None
        for line in haystack.splitlines():
            s = line.strip()
            if s.lower().startswith("question:"):
                question = s.split(":", 1)[1].strip()
                break
        if not question:
            return
        self.chat_panel.append_system_note(
            "agent has a question from its last decision", color="#ffcc66"
        )
        # Record as if the agent itself spoke in chat, so the operator's reply
        # naturally continues the thread.
        self.chat.add_assistant(f"QUESTION: {question}")
        self.chat_panel.begin_assistant()
        self.chat_panel.append_assistant_delta(f"QUESTION: {question}")
        self.chat_panel.end_assistant()
        if self.right_tabs.currentIndex() != self._chat_tab_index:
            self._start_chat_flash()

    # ----- chat-tab attention flash -----------------------------------------
    def _start_chat_flash(self) -> None:
        if not self._chat_flash_timer.isActive():
            self._chat_flash_on = False
            self._chat_flash_timer.start()

    def _stop_chat_flash(self) -> None:
        self._chat_flash_timer.stop()
        self._chat_flash_on = False
        bar = self.right_tabs.tabBar()
        if bar is not None:
            bar.setTabTextColor(self._chat_tab_index, QColor())
        self.right_tabs.setTabText(self._chat_tab_index, "Chat with agent")

    def _tick_chat_flash(self) -> None:
        bar = self.right_tabs.tabBar()
        if bar is None:
            return
        self._chat_flash_on = not self._chat_flash_on
        if self._chat_flash_on:
            bar.setTabTextColor(self._chat_tab_index, QColor("#ff8800"))
            self.right_tabs.setTabText(self._chat_tab_index, "● Chat with agent")
        else:
            bar.setTabTextColor(self._chat_tab_index, QColor("#cc6600"))
            self.right_tabs.setTabText(self._chat_tab_index, "Chat with agent")

    def _on_right_tab_changed(self, index: int) -> None:
        if index == self._chat_tab_index:
            self._stop_chat_flash()

    def _on_send_proposed(self) -> None:
        text = self.proposed_edit.text().strip()
        if not text:
            return
        self._send_command(text, source="agent")
        self._finalize_trace(text, approved=True)
        self.proposed_edit.clear()
        # Proactive mode: schedule a follow-up decision even if the MUD is
        # silent. Any incoming MUD text simply restarts this same timer, so
        # the debounce semantics still apply when the MUD is talking.
        if self.cfg.agent.proactive_decisions:
            self._idle_timer.start(max(50, self.cfg.mud.decision_idle_ms))

    def _on_reject_proposed(self) -> None:
        text = self.proposed_edit.text().strip()
        self.reasoning_view.append_note(f"[rejected] {text!r}", color="#f88")
        self._finalize_trace(text, approved=False)
        self.proposed_edit.clear()
        if self.cfg.agent.proactive_decisions:
            self._idle_timer.start(max(50, self.cfg.mud.decision_idle_ms))

    def _finalize_trace(self, command: str, approved: bool) -> None:
        if self._pending_decision is None or self._pending_last_log is None:
            return
        # Defer outcome capture briefly so MUD response is included.
        decision = self._pending_decision
        base = self._pending_last_log
        self._pending_decision = None
        self._pending_last_log = None

        def write_trace() -> None:
            outcome = "".join(self._pending_outcome_buf)
            self._pending_outcome_buf = []
            self.trace_logger.log({
                **base,
                "reasoning": decision.reasoning,
                "thinking": decision.thinking,
                "raw_response": decision.raw,
                "command": command,
                "approved": approved,
                "outcome": outcome,
            })
        QTimer.singleShot(2000, write_trace)

    # ----- lifecycle ---------------------------------------------------------
    def closeEvent(self, a0: Any) -> None:  # type: ignore[override]
        if self.mud is not None:
            try:
                asyncio.ensure_future(self.mud.close())
            except Exception:
                pass
        self.cfg.save()
        super().closeEvent(a0)
