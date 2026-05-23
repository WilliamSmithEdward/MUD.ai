"""Settings dialog: connection, model, sampling, context window, autonomy."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..llm import models_catalog


class SettingsDialog(QDialog):
    def __init__(self, cfg: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MUD.ai Settings")
        self.cfg = cfg

        tabs = QTabWidget(self)

        # --- MUD tab ---------------------------------------------------------
        mud_tab = QWidget()
        mud_form = QFormLayout(mud_tab)
        self.host_edit = QLineEdit(cfg.mud.host)
        self.port_spin = QSpinBox(); self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(cfg.mud.port)
        self.encoding_edit = QLineEdit(cfg.mud.encoding)
        self.idle_spin = QSpinBox(); self.idle_spin.setRange(100, 10000)
        self.idle_spin.setValue(cfg.mud.decision_idle_ms)
        self.idle_spin.setSuffix(" ms")
        self.auto_connect_chk = QCheckBox("Auto-connect on startup")
        self.auto_connect_chk.setChecked(cfg.mud.auto_connect_on_start)
        mud_form.addRow("Host:", self.host_edit)
        mud_form.addRow("Port:", self.port_spin)
        mud_form.addRow("Encoding:", self.encoding_edit)
        mud_form.addRow("Decision idle delay:", self.idle_spin)
        mud_form.addRow("", self.auto_connect_chk)
        tabs.addTab(mud_tab, "MUD")

        # --- LLM tab ---------------------------------------------------------
        llm_tab = QWidget()
        llm_form = QFormLayout(llm_tab)
        self.model_combo = QComboBox()
        for entry in models_catalog.CATALOG:
            self.model_combo.addItem(entry.label, entry.filename)
        # Allow custom filename if not in catalog.
        idx = self.model_combo.findData(cfg.llm.model_file)
        if idx < 0:
            self.model_combo.addItem(f"(custom) {cfg.llm.model_file}", cfg.llm.model_file)
            idx = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(idx)

        self.ctx_spin = QSpinBox(); self.ctx_spin.setRange(1024, 131072)
        self.ctx_spin.setSingleStep(1024); self.ctx_spin.setValue(cfg.llm.n_ctx)
        self.gpu_layers_spin = QSpinBox(); self.gpu_layers_spin.setRange(-1, 999)
        self.gpu_layers_spin.setValue(cfg.llm.n_gpu_layers)
        self.gpu_layers_spin.setSpecialValueText("all (-1)")
        self.threads_spin = QSpinBox(); self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(cfg.llm.n_threads)
        self.transcript_budget_spin = QSpinBox()
        self.transcript_budget_spin.setRange(512, 100000)
        self.transcript_budget_spin.setSingleStep(256)
        self.transcript_budget_spin.setValue(cfg.llm.transcript_token_budget)
        self.max_decision_spin = QSpinBox()
        self.max_decision_spin.setRange(32, 4096)
        self.max_decision_spin.setValue(cfg.llm.max_decision_tokens)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0); self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(cfg.llm.temperature)
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0); self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(cfg.llm.top_p)
        self.top_k_spin = QSpinBox(); self.top_k_spin.setRange(0, 1000)
        self.top_k_spin.setValue(cfg.llm.top_k)
        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(0.5, 2.0)
        self.repeat_penalty_spin.setSingleStep(0.01)
        self.repeat_penalty_spin.setValue(cfg.llm.repeat_penalty)

        llm_form.addRow("Model:", self.model_combo)
        llm_form.addRow("Context window (n_ctx):", self.ctx_spin)
        llm_form.addRow("Transcript token budget:", self.transcript_budget_spin)
        llm_form.addRow("Max decision tokens:", self.max_decision_spin)
        llm_form.addRow("GPU layers:", self.gpu_layers_spin)
        llm_form.addRow("CPU threads:", self.threads_spin)
        llm_form.addRow("Temperature:", self.temp_spin)
        llm_form.addRow("top_p:", self.top_p_spin)
        llm_form.addRow("top_k:", self.top_k_spin)
        llm_form.addRow("Repeat penalty:", self.repeat_penalty_spin)
        tabs.addTab(llm_tab, "LLM")

        # --- Agent tab -------------------------------------------------------
        agent_tab = QWidget()
        agent_form = QFormLayout(agent_tab)
        self.auto_send_chk = QCheckBox("Auto-send LLM commands")
        self.auto_send_chk.setChecked(cfg.agent.auto_send)
        self.auto_load_chk = QCheckBox("Auto-load model on startup")
        self.auto_load_chk.setChecked(cfg.agent.auto_load_model_on_start)
        self.proactive_chk = QCheckBox(
            "Proactive decisions (keep thinking even when MUD is silent)"
        )
        self.proactive_chk.setToolTip(
            "On: after every send/reject the agent schedules its own next "
            "decision via the idle timer. Off: decisions only fire when the "
            "MUD sends new text (purely reactive)."
        )
        self.proactive_chk.setChecked(cfg.agent.proactive_decisions)
        self.min_interval_spin = QSpinBox()
        self.min_interval_spin.setRange(0, 30000)
        self.min_interval_spin.setSingleStep(100)
        self.min_interval_spin.setValue(cfg.agent.min_command_interval_ms)
        self.min_interval_spin.setSuffix(" ms")
        agent_form.addRow("Default autonomy:", self.auto_send_chk)
        agent_form.addRow("", self.auto_load_chk)
        agent_form.addRow("", self.proactive_chk)
        agent_form.addRow("Min interval between auto-sends:", self.min_interval_spin)
        tabs.addTab(agent_tab, "Agent")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)
        del Qt  # appease unused-import

    def apply_to(self, cfg: AppConfig) -> None:
        cfg.mud.host = self.host_edit.text().strip() or cfg.mud.host
        cfg.mud.port = self.port_spin.value()
        cfg.mud.encoding = self.encoding_edit.text().strip() or "utf-8"
        cfg.mud.decision_idle_ms = self.idle_spin.value()
        cfg.mud.auto_connect_on_start = self.auto_connect_chk.isChecked()

        filename = self.model_combo.currentData()
        if isinstance(filename, str) and filename:
            cfg.llm.model_file = filename
        cfg.llm.n_ctx = self.ctx_spin.value()
        cfg.llm.n_gpu_layers = self.gpu_layers_spin.value()
        cfg.llm.n_threads = self.threads_spin.value()
        cfg.llm.transcript_token_budget = self.transcript_budget_spin.value()
        cfg.llm.max_decision_tokens = self.max_decision_spin.value()
        cfg.llm.temperature = self.temp_spin.value()
        cfg.llm.top_p = self.top_p_spin.value()
        cfg.llm.top_k = self.top_k_spin.value()
        cfg.llm.repeat_penalty = self.repeat_penalty_spin.value()

        cfg.agent.auto_send = self.auto_send_chk.isChecked()
        cfg.agent.auto_load_model_on_start = self.auto_load_chk.isChecked()
        cfg.agent.proactive_decisions = self.proactive_chk.isChecked()
        cfg.agent.min_command_interval_ms = self.min_interval_spin.value()
