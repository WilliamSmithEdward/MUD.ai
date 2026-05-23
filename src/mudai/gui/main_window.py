"""Main application window: wires MUD client, agent, GUI, and decision loop."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
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
from ..llm.experience import TraceIndex
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
        from ..config import MEMORY_PATH, TRACES_DIR
        self.memory = MemoryStore.load(MEMORY_PATH)
        # Experience index: past approved decisions, used for in-context
        # examples so the agent improves as more sessions accumulate.
        self.experience = TraceIndex(TRACES_DIR)
        try:
            indexed = self.experience.reload()
        except Exception:
            indexed = 0
        self.agent = Agent(
            self.backend, cfg.agent, cfg.llm,
            memory=self.memory, experience=self.experience,
        )
        self.chat = ChatSession(self.backend, self.agent, cfg.agent, cfg.llm)
        self.mud: MudClient | None = None
        self.trace_logger = TraceLogger()
        self._decision_inflight = False
        self._reflect_inflight = False
        self._approved_since_reflect = 0
        self._initial_indexed_count = indexed
        self._chat_inflight = False
        self._last_send_ms = 0.0
        # Loop-resilience counters.
        self._empty_proposals_in_row = 0
        self._decision_errors_in_row = 0
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

        if self._initial_indexed_count:
            self.reasoning_view.append_note(
                f"[experience] indexed {self._initial_indexed_count} past"
                f" approved decision(s); top-"
                f"{self.cfg.agent.experience_examples_k} will be injected into"
                " each decision prompt.",
                color="#8af",
            )

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

        # One-shot operator note input. Goes into the transcript as
        # [OPERATOR] context for the next decision (NOT sent to the MUD).
        self.note_edit = QLineEdit()
        self.note_edit.setFont(mono_font())
        self.note_edit.setPlaceholderText(
            "Inject one-shot note to agent (Enter to send; not sent to MUD)"
        )
        self.note_edit.returnPressed.connect(self._on_note_edit_submit)
        self.note_send_btn = QPushButton("Inject")
        self.note_send_btn.clicked.connect(self._on_note_edit_submit)
        note_row = QHBoxLayout()
        note_row.setContentsMargins(0, 0, 0, 0)
        note_row.addWidget(QLabel("Note:"))
        note_row.addWidget(self.note_edit, 1)
        note_row.addWidget(self.note_send_btn)
        note_container = QWidget()
        nc_lay = QVBoxLayout(note_container)
        nc_lay.setContentsMargins(0, 0, 0, 0)
        nc_lay.addLayout(note_row)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self.reasoning_view)
        right_split.addWidget(note_container)
        right_split.setStretchFactor(0, 10)
        right_split.setStretchFactor(1, 0)

        self.chat_panel = ChatPanel()
        self.chat_panel.message_submitted.connect(self._on_chat_user_submit)
        self.chat_panel.clear_requested.connect(self._on_chat_clear)

        self.memory_panel = MemoryPanel(self.memory)

        # Reasoning is always visible on top; chat, memory, and steering
        # notes live in a tabbed pane below it so the agent's gameloop is
        # never hidden.
        right_tabs = QTabWidget()
        self._chat_tab_index = right_tabs.addTab(self.chat_panel, "Chat with agent")
        right_tabs.addTab(self.memory_panel, "Memory")
        right_tabs.addTab(steering_container, "Steering")
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
        self.proposed_edit.setPlaceholderText("Proposed command(s) from LLM. Separate multiple commands with ' ; '. Editable.")
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
        self.manual_edit.setPlaceholderText("Manual command (Enter to send; separate multiple with ' ; '; empty = bare CR)")
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

        # Decision-window dropdown: how long to wait (idle) before the agent
        # makes its next decision. Editable so operator can type any value.
        tb.addWidget(QLabel(" Decision window: "))
        self.decision_window_combo = QComboBox()
        self.decision_window_combo.setEditable(True)
        self.decision_window_combo.setInsertPolicy(
            QComboBox.InsertPolicy.NoInsert
        )
        self._decision_window_presets_ms: list[int] = [
            250, 500, 1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000,
        ]
        for ms in self._decision_window_presets_ms:
            self.decision_window_combo.addItem(self._fmt_window(ms), ms)
        self._sync_decision_window_combo()
        self.decision_window_combo.currentIndexChanged.connect(
            self._on_decision_window_index_changed
        )
        line_edit = self.decision_window_combo.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(
                self._on_decision_window_edit_finished
            )
        self.decision_window_combo.setToolTip(
            "Idle delay after MUD text (or after a sent command in proactive "
            "mode) before the agent thinks again."
        )
        tb.addWidget(self.decision_window_combo)

        tb.addSeparator()

        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(self._on_settings_clicked)
        tb.addAction(act_settings)

        self.act_reflect = QAction("Reflect", self)
        self.act_reflect.setToolTip(
            "Have the LLM review recent approved decisions and distill durable"
            " lessons into permanent memory."
        )
        self.act_reflect.triggered.connect(self._on_reflect_clicked)
        tb.addAction(self.act_reflect)

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

    # ----- decision window dropdown -----------------------------------------
    @staticmethod
    def _fmt_window(ms: int) -> str:
        if ms < 1000:
            return f"{ms} ms"
        s = ms / 1000.0
        return f"{s:.2f}".rstrip("0").rstrip(".") + " s"

    def _sync_decision_window_combo(self) -> None:
        """Reflect cfg.mud.decision_idle_ms in the toolbar dropdown."""
        cur_ms = int(self.cfg.mud.decision_idle_ms)
        combo = self.decision_window_combo
        combo.blockSignals(True)
        # Find or add an item matching the current value.
        idx = -1
        for i in range(combo.count()):
            if int(combo.itemData(i)) == cur_ms:
                idx = i
                break
        if idx < 0:
            combo.addItem(self._fmt_window(cur_ms), cur_ms)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)
        line_edit = combo.lineEdit()
        if line_edit is not None:
            line_edit.setText(self._fmt_window(cur_ms))
        combo.blockSignals(False)

    def _apply_decision_window_ms(self, ms: int) -> None:
        ms = max(50, min(60000, int(ms)))
        if ms == int(self.cfg.mud.decision_idle_ms):
            return
        self.cfg.mud.decision_idle_ms = ms
        # If the idle timer is currently pending, restart it with the new delay.
        if self._idle_timer.isActive():
            self._idle_timer.start(ms)
        self.reasoning_view.append_note(
            f"[decision window] set to {self._fmt_window(ms)}", color="#888"
        )

    def _on_decision_window_index_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self.decision_window_combo.itemData(index)
        if data is None:
            return
        self._apply_decision_window_ms(int(data))

    def _on_decision_window_edit_finished(self) -> None:
        line_edit = self.decision_window_combo.lineEdit()
        if line_edit is None:
            return
        text = line_edit.text().strip().lower()
        # Parse "2500 ms", "2.5 s", "2500", etc.
        try:
            if text.endswith("ms"):
                ms = int(float(text[:-2].strip()))
            elif text.endswith("s"):
                ms = int(float(text[:-1].strip()) * 1000)
            else:
                # Heuristic: bare number <= 60 -> seconds, otherwise ms.
                val = float(text)
                ms = int(val * 1000) if val <= 60 else int(val)
        except ValueError:
            self._sync_decision_window_combo()
            return
        self._apply_decision_window_ms(ms)
        self._sync_decision_window_combo()

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
        xp = f"experience={len(self.experience.entries)} past decisions"
        sb.showMessage(f"{conn}   |   {model}   |   {ctx}   |   {xp}")

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
        was_connected = self.act_disconnect.isEnabled()
        is_conn = bool(self.mud and self.mud.connected)
        self.act_connect.setEnabled(not is_conn)
        self.act_disconnect.setEnabled(is_conn)
        # On transition connected -> disconnected, optionally auto-reflect so
        # lessons from this session are saved to memory before the operator
        # closes the app.
        if was_connected and not is_conn:
            if (
                getattr(self.cfg.agent, "reflect_on_disconnect", False)
                and self.backend.loaded
                and not self._reflect_inflight
            ):
                self._start_reflection(reason="disconnect")

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
        # If auto-send is on, jump straight in: kick the idle timer so the
        # agent makes its first decision without waiting for fresh MUD text.
        # This handles the common case where the operator connects first,
        # then loads the model -- the transcript already has plenty of
        # context to act on.
        if self.cfg.agent.auto_send and not self._decision_inflight:
            self._idle_timer.start(max(50, self.cfg.mud.decision_idle_ms))

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
            self._sync_decision_window_combo()

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
        # Allow an empty line: many MUDs use a bare CR to page output,
        # dismiss prompts, or trigger the "more" pager. We must NOT swallow
        # the keystroke.
        self.manual_edit.clear()
        if text:
            # Only non-empty commands go into history.
            if not self._cmd_history or self._cmd_history[-1] != text:
                self._cmd_history.append(text)
                if len(self._cmd_history) > 500:
                    self._cmd_history = self._cmd_history[-500:]
        self._cmd_history_idx = len(self._cmd_history)
        # Support `;`-separated chains, same syntax as the agent's proposed
        # field. Empty input -> single bare CR send (see above).
        cmds = Agent.split_commands(text) if text.strip() else [text]
        if not cmds:
            cmds = [text]
        self._send_command(cmds[0], source="you")
        gap = max(50, self.cfg.agent.min_command_interval_ms)
        for i, cmd in enumerate(cmds[1:], start=1):
            QTimer.singleShot(i * gap, lambda c=cmd: self._send_command(c, source="you"))

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

    def _on_note_edit_submit(self) -> None:
        text = self.note_edit.text().strip()
        if not text:
            return
        self.agent.add_operator_note(text)
        self.reasoning_view.append_note(f"[operator note] {text}", color="#ffcc66")
        self.note_edit.clear()

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
        self._decision_errors_in_row = 0
        self._pending_decision = decision
        self._pending_outcome_buf = []
        self.proposed_edit.setText(decision.command)
        self.reasoning_view.append_note(
            f"[proposed] {decision.command!r}", color="#ffcc66"
        )
        # Track empty proposals; after 2 in a row, inject a one-shot nudge so
        # the next decision is forced toward emitting a COMMAND: line instead
        # of looping on the same stuck state. This is what stops the agent
        # from getting "stuck" producing empty proposals indefinitely.
        if not decision.command.strip():
            self._empty_proposals_in_row += 1
            if self._empty_proposals_in_row >= 2:
                nudge = (
                    "Your previous response had no `COMMAND:` line. Stop"
                    " thinking and on the next turn output exactly one line"
                    " starting with `COMMAND: ` followed by a single safe"
                    " MUD command (e.g. `COMMAND: look`)."
                )
                self.agent.add_operator_note(nudge)
                self.reasoning_view.append_note(
                    f"[nudge] {nudge}", color="#fa6"
                )
                self._empty_proposals_in_row = 0
        else:
            self._empty_proposals_in_row = 0
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
        elif self.cfg.agent.proactive_decisions:
            # auto_send is off and the operator hasn't reviewed yet, but the
            # proposal text was emitted. We do NOT reschedule here -- the loop
            # rearms when the operator clicks Send or Reject. That's the
            # intended manual-review flow.
            pass

    def _on_decision_error(self, msg: str) -> None:
        self._decision_inflight = False
        self.reasoning_view.append_note(f"[llm error] {msg}", color="#f88")
        # Don't let one bad inference kill the loop. Back off with an
        # exponential-ish delay so we don't spin on a hard failure (e.g.,
        # OOM, model unloaded mid-stream) but still recover automatically.
        self._decision_errors_in_row += 1
        if (
            self.cfg.agent.proactive_decisions
            and self.backend.loaded
            and self.mud is not None
            and self.mud.connected
        ):
            backoff = min(30000, 1000 * (2 ** min(5, self._decision_errors_in_row - 1)))
            self._idle_timer.start(backoff)

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
            # Model produced no parseable command this turn. Don't stall: drop
            # the empty proposal, capture the (empty) trace as unapproved so it
            # doesn't bias future retrieval, and immediately schedule another
            # decision so the loop keeps going.
            self.reasoning_view.append_note(
                "[skip] empty proposal - rescheduling next decision",
                color="#fa6",
            )
            if self._pending_decision is not None:
                self._finalize_trace("", approved=False)
            self.proposed_edit.clear()
            # Kick the idle timer regardless of proactive_decisions: an empty
            # proposal is a dead-end and we must keep the loop alive.
            self._idle_timer.start(max(50, self.cfg.mud.decision_idle_ms))
            return
        # The model may have emitted multiple commands separated by " ; "
        # (or the operator may have edited the proposal to include `;`).
        # Send them in order with min_command_interval spacing so the MUD
        # has time to process each one.
        cmds = Agent.split_commands(text)
        if not cmds:
            cmds = [text]
        self._send_command(cmds[0], source="agent")
        # Queue the remainder via single-shot QTimers so the GUI stays
        # responsive and the MUD isn't flooded.
        gap = max(50, self.cfg.agent.min_command_interval_ms)
        for i, cmd in enumerate(cmds[1:], start=1):
            QTimer.singleShot(i * gap, lambda c=cmd: self._send_command(c, source="agent"))
        # For the trace, record the joined form so future retrieval can see
        # the whole chain as one decision.
        joined = " ; ".join(cmds)
        self._finalize_trace(joined, approved=True)
        self.proposed_edit.clear()
        # Proactive mode: schedule a follow-up decision after the LAST queued
        # command has had a chance to fire. Any incoming MUD text simply
        # restarts this same timer.
        if self.cfg.agent.proactive_decisions:
            tail_delay = (len(cmds) - 1) * gap if len(cmds) > 1 else 0
            self._idle_timer.start(max(50, self.cfg.mud.decision_idle_ms + tail_delay))

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
            row = {
                **base,
                "reasoning": decision.reasoning,
                "thinking": decision.thinking,
                "raw_response": decision.raw,
                "command": command,
                "approved": approved,
                "outcome": outcome,
            }
            self.trace_logger.log(row)
            # Feed into the in-process experience index so the next decision
            # can already retrieve this example.
            try:
                if approved:
                    self.experience.add_row(row)
                    self._approved_since_reflect += 1
                    self._maybe_auto_reflect()
            except Exception:
                pass
        QTimer.singleShot(2000, write_trace)

    # ----- reflection (self-distillation into memory) -----------------------
    def _recent_trace_examples(self, n: int = 12) -> list[str]:
        """Return rendered text blocks of the most recent approved decisions."""
        if not self.experience.entries:
            return []
        # Take the tail (most recently appended -> most recent in time).
        tail = self.experience.entries[-n:]
        return [
            TraceIndex.render_examples([e]) for e in tail if e.command
        ]

    def _maybe_auto_reflect(self) -> None:
        every = int(getattr(self.cfg.agent, "reflect_every_n_decisions", 0) or 0)
        if every <= 0:
            return
        if self._approved_since_reflect < every:
            return
        if self._reflect_inflight or not self.backend.loaded:
            return
        self._approved_since_reflect = 0
        self._start_reflection(reason=f"auto (every {every} approvals)")

    def _on_reflect_clicked(self) -> None:
        if not self.backend.loaded:
            QMessageBox.information(
                self, "Reflect", "Load the LLM first (toolbar toggle)."
            )
            return
        if self._reflect_inflight:
            return
        self._start_reflection(reason="manual")

    def _start_reflection(self, reason: str) -> None:
        examples = self._recent_trace_examples(n=12)
        if not examples:
            self.reasoning_view.append_note(
                "[reflect] no recent approved decisions yet; skipping.",
                color="#888",
            )
            return
        self._reflect_inflight = True
        self.reasoning_view.append_note(
            f"[reflect] reviewing {len(examples)} recent decisions "
            f"({reason}) ...",
            color="#8af",
        )
        asyncio.ensure_future(self._reflect_task(examples))

    async def _reflect_task(self, examples: list[str]) -> None:
        try:
            raw = await asyncio.to_thread(self.agent.reflect, examples, 5)
        except Exception as e:
            self.reasoning_view.append_note(
                f"[reflect] failed: {e}", color="#f88"
            )
            self._reflect_inflight = False
            return
        # Persist any REMEMBER: lines into permanent memory.
        added: list[Any] = []
        try:
            added = self.memory.capture_from_text(raw, source="agent_decision")
        except Exception:
            added = []
        if added:
            self.reasoning_view.append_note(
                f"[reflect] saved {len(added)} new lesson(s) to memory.",
                color="#8f8",
            )
        else:
            self.reasoning_view.append_note(
                "[reflect] no new durable lessons this pass.", color="#888"
            )
        self._reflect_inflight = False

    # ----- lifecycle ---------------------------------------------------------
    def closeEvent(self, a0: Any) -> None:  # type: ignore[override]
        if self.mud is not None:
            try:
                asyncio.ensure_future(self.mud.close())
            except Exception:
                pass
        self.cfg.save()
        super().closeEvent(a0)
