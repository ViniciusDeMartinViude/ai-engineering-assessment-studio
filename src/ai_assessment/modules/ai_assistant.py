from __future__ import annotations

import hashlib
import os
import uuid

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.workspace import CandidateWorkspace
from ..services.ai_context import AIContextBuffer
from ..services.ai_gateway import GatewayClient, GatewayClientError, SelectedContext
from ..services.ai_ide_transfer import AIIdeCoordinator, AIAskRequest, CodeBlock, IDEHandoff, extract_code_blocks


class _GatewayWorker(QObject):
    completed = Signal(str, object)
    failed = Signal(str, object)

    def __init__(self, operation: str, client: GatewayClient, *, message: str = "", context: SelectedContext | None = None, key: str = "") -> None:
        super().__init__()
        self.operation = operation
        self.client = client
        self.message = message
        self.context = context
        self.key = key

    def run(self) -> None:
        try:
            if self.operation == "send":
                result = self.client.send_message(self.message, self.context or SelectedContext.create("selected_text", ""), idempotency_key=self.key, request_id=self.key)
            elif self.operation == "allowance":
                result = self.client.allowance()
            else:
                result = {"items": self.client.history()}
            self.completed.emit(self.operation, result)
        except Exception as exc:
            self.failed.emit(self.operation, exc)


class AIAssistantPage(QWidget):
    def __init__(self, workspace: CandidateWorkspace, context_buffer: AIContextBuffer | None = None, coordinator: AIIdeCoordinator | None = None) -> None:
        super().__init__()
        self.workspace = workspace
        self.context_buffer = context_buffer or AIContextBuffer()
        self.coordinator = coordinator
        if self.coordinator is not None:
            self.coordinator.ask_requested.connect(self._receive_ide_request)
        self._thread: QThread | None = None
        self._worker: _GatewayWorker | None = None
        self._refresh_history_after_worker = False
        self._pending: tuple[str, SelectedContext, str] | None = None
        self._ask_origin: AIAskRequest | None = None
        self._response_origin: AIAskRequest | None = None
        self._current_response_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(10)
        heading = QHBoxLayout()
        title = QLabel("AI Assistant")
        title.setObjectName("pageTitle")
        heading.addWidget(title)
        heading.addStretch(1)
        root.addLayout(heading)
        summary = QLabel("Organizer gateway only. Attach context deliberately; the desktop never holds an OpenAI key and is not a security boundary.")
        summary.setObjectName("summary")
        summary.setWordWrap(True)
        root.addWidget(summary)

        connection = QFrame()
        connection.setObjectName("aiConnectionPanel")
        form = QFormLayout(connection)
        self.gateway_url = QLineEdit(os.environ.get("AI_GATEWAY_URL", "http://127.0.0.1:8765"))
        self.credential = QLineEdit(os.environ.get("AI_GATEWAY_SESSION_TOKEN", ""))
        self.credential.setEchoMode(QLineEdit.EchoMode.Password)
        self.credential.setPlaceholderText("practice session credential from the gateway operator")
        form.addRow("Gateway URL", self.gateway_url)
        form.addRow("Session credential", self.credential)
        connection_buttons = QHBoxLayout()
        self.connect_button = QPushButton("Connect / refresh")
        self.connect_button.clicked.connect(self._refresh_gateway)
        connection_buttons.addWidget(self.connect_button)
        self.connection_status = QLabel("Not connected")
        connection_buttons.addWidget(self.connection_status)
        connection_buttons.addStretch(1)
        self.model_label = QLabel("Model: unavailable")
        connection_buttons.addWidget(self.model_label)
        self.allowance_label = QLabel("Allowance: unavailable")
        connection_buttons.addWidget(self.allowance_label)
        form.addRow("", connection_buttons)
        root.addWidget(connection)

        split = QSplitter(Qt.Orientation.Vertical)
        conversation = QFrame()
        conversation_layout = QVBoxLayout(conversation)
        conversation_layout.addWidget(QLabel("Conversation history"))
        self.history_view = QPlainTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setMaximumBlockCount(2000)
        conversation_layout.addWidget(self.history_view)
        conversation_layout.addWidget(QLabel("Generated code blocks"))
        self.code_blocks_scroll = QScrollArea()
        self.code_blocks_scroll.setWidgetResizable(True)
        self.code_blocks_container = QWidget()
        self.code_blocks_layout = QVBoxLayout(self.code_blocks_container)
        self.code_blocks_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.code_blocks_scroll.setWidget(self.code_blocks_container)
        conversation_layout.addWidget(self.code_blocks_scroll)
        split.addWidget(conversation)

        compose = QFrame()
        compose_layout = QVBoxLayout(compose)
        compose_layout.addWidget(QLabel("New request"))
        self.message = QPlainTextEdit()
        self.message.setPlaceholderText("Ask a question for the approved organizer model...")
        self.message.setMaximumHeight(110)
        compose_layout.addWidget(self.message)
        context_bar = QHBoxLayout()
        self.context_type = QComboBox()
        self.context_type.addItem("Selected IDE text", "selected_text")
        self.context_type.addItem("Error traceback", "error_traceback")
        self.context_type.addItem("Training metrics", "training_metrics")
        context_bar.addWidget(QLabel("Attached context:"))
        context_bar.addWidget(self.context_type)
        use_ide = QPushButton("Use IDE selection")
        use_ide.clicked.connect(self._use_ide_context)
        context_bar.addWidget(use_ide)
        context_bar.addStretch(1)
        compose_layout.addLayout(context_bar)
        self.context_preview = QPlainTextEdit()
        self.context_preview.setPlaceholderText("Nothing will be sent until you add context here. The preview is exactly what will be attached.")
        self.context_preview.setMaximumHeight(120)
        compose_layout.addWidget(self.context_preview)
        send_bar = QHBoxLayout()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send)
        send_bar.addWidget(self.send_button)
        self.retry_button = QPushButton("Retry same request")
        self.retry_button.setVisible(False)
        self.retry_button.clicked.connect(self._retry)
        send_bar.addWidget(self.retry_button)
        self.request_status = QLabel("Ready")
        send_bar.addWidget(self.request_status)
        send_bar.addStretch(1)
        compose_layout.addLayout(send_bar)
        split.addWidget(compose)
        split.setSizes([420, 300])
        root.addWidget(split, 1)
        self.setStyleSheet(
            """
            #aiConnectionPanel, #aiConnectionPanel QLineEdit, #aiConnectionPanel QComboBox, #aiConnectionPanel QPlainTextEdit {
                border: 1px solid #b7c8bf;
                border-radius: 5px;
            }
            """
        )

    def _client(self) -> GatewayClient:
        return GatewayClient(self.gateway_url.text().strip(), self.credential.text())

    def _start_worker(self, worker: _GatewayWorker) -> None:
        if self._thread is not None:
            return
        self._worker = worker
        self._thread = QThread(self)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.completed.connect(self._worker_completed)
        worker.failed.connect(self._worker_failed)
        worker.completed.connect(self._finish_worker)
        worker.failed.connect(self._finish_worker)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._worker_thread_finished)
        self._thread.start()

    def _finish_worker(self, _operation: str, _value: object) -> None:
        if self._thread is not None:
            self._thread.quit()

    def _worker_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        if self._refresh_history_after_worker:
            self._refresh_history_after_worker = False
            self._start_history_refresh()

    def _refresh_gateway(self) -> None:
        if not self.credential.text().strip():
            self.connection_status.setText("Enter a server-provisioned practice credential")
            return
        self.connect_button.setEnabled(False)
        self.connection_status.setText("Connecting...")
        self._start_worker(_GatewayWorker("allowance", self._client()))

    def _worker_completed(self, operation: str, value: object) -> None:
        data = value if isinstance(value, dict) else {}
        if operation == "allowance":
            self.connection_status.setText("Connected")
            self.model_label.setText(f"Model: {data.get('model', 'unavailable')}")
            self.allowance_label.setText(f"Allowance: {data.get('remaining_cents', 'unavailable')} cents")
            self.connect_button.setEnabled(True)
            self._refresh_history_after_worker = True
        elif operation == "history":
            self._render_history(data.get("items", []))
        elif operation == "send":
            self._pending = None
            self._response_origin = self._ask_origin
            self._ask_origin = None
            self.retry_button.setVisible(False)
            self.send_button.setEnabled(True)
            self.request_status.setText("Response received")
            answer = data.get("answer") or "(No response text returned)"
            self.history_view.appendPlainText(f"Assistant: {answer}\n")
            self.allowance_label.setText(f"Allowance: {data.get('remaining_cents', 'unavailable')} cents")
            self._current_response_id = str(data.get("request_id") or "")
            self._render_code_blocks(answer, self._current_response_id)
            self.workspace.record_event("ai.result", {"request_id": data.get("request_id"), "status": data.get("status"), "response_chars": len(answer), "response_sha256": hashlib.sha256(answer.encode('utf-8')).hexdigest(), "remaining_cents": data.get("remaining_cents")}, outcome=data.get("status", "completed"))

    def _worker_failed(self, operation: str, error: object) -> None:
        self.connect_button.setEnabled(True)
        self.send_button.setEnabled(True)
        if isinstance(error, GatewayClientError):
            message = f"{error.code}: {error.message}"
        else:
            message = str(error)
        self.connection_status.setText("Gateway error")
        self.request_status.setText(message)
        if operation == "send":
            error_code = error.code if isinstance(error, GatewayClientError) else "client_error"
            request_id = self._pending[2] if self._pending is not None else None
            self.workspace.record_event("ai.result", {"request_id": request_id, "status": "failed", "error_code": error_code}, outcome="failed")
            self.retry_button.setVisible(self._pending is not None)
        else:
            QMessageBox.warning(self, "AI gateway", message)

    def _start_history_refresh(self) -> None:
        if self._thread is None:
            self._start_worker(_GatewayWorker("history", self._client()))

    def _render_history(self, items: list[dict[str, object]]) -> None:
        self.history_view.clear()
        for item in reversed(items):
            self.history_view.appendPlainText(f"Request {item.get('request_id')} [{item.get('status')}]\nAssistant: {item.get('response_text') or '(no response)'}\n")

    def _use_ide_context(self) -> None:
        if not self.context_buffer.text:
            self.request_status.setText("No IDE selection has been attached")
            return
        index = self.context_type.findData(self.context_buffer.kind)
        if index >= 0:
            self.context_type.setCurrentIndex(index)
        self.context_preview.setPlainText(self.context_buffer.text)
        self.request_status.setText(f"Attached {self.context_buffer.source or 'IDE'} context")

    def _receive_ide_request(self, request: AIAskRequest) -> None:
        self._ask_origin = request
        self.message.setPlainText(request.question)
        selected_index = self.context_type.findData("selected_text")
        if selected_index >= 0:
            self.context_type.setCurrentIndex(selected_index)
        self.context_preview.setPlainText(request.code)
        self.request_status.setText("IDE request preview ready. Review the question and code, then press Send.")
        self.message.setFocus()

    def _render_code_blocks(self, answer: str, response_id: str) -> None:
        while self.code_blocks_layout.count():
            item = self.code_blocks_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        blocks = extract_code_blocks(answer)
        if not blocks:
            self.code_blocks_layout.addWidget(QLabel("No fenced code blocks were returned."))
            return
        for block in blocks:
            panel = QFrame()
            panel.setFrameShape(QFrame.Shape.StyledPanel)
            panel_layout = QVBoxLayout(panel)
            panel_layout.addWidget(QLabel(f"{block.block_id} · {block.language} · {len(block.code)} characters"))
            preview = QPlainTextEdit()
            preview.setReadOnly(True)
            preview.setPlainText(block.code)
            preview.setMaximumHeight(130)
            panel_layout.addWidget(preview)
            send_button = QPushButton("Send to IDE")
            send_button.clicked.connect(lambda _checked=False, b=block, rid=response_id: self._send_block_to_ide(rid, b))
            panel_layout.addWidget(send_button)
            self.code_blocks_layout.addWidget(panel)

    def _send_block_to_ide(self, response_id: str, block: CodeBlock) -> None:
        if self.coordinator is None:
            self.request_status.setText("IDE handoff is unavailable")
            return
        origin = self._response_origin if self._current_response_id == response_id else None
        self.coordinator.request_ide_handoff(
            IDEHandoff(
                response_id=response_id,
                block=block,
                source_path=origin.source_path if origin else None,
                source_file_hash=origin.file_hash if origin else None,
                source_selection_hash=origin.selection_hash if origin else None,
            )
        )
        self.request_status.setText(f"Selected {block.block_id} for IDE handoff")

    def _send(self) -> None:
        self._begin_send()

    def _retry(self) -> None:
        self._begin_send(retry=True)

    def _begin_send(self, *, retry: bool = False) -> None:
        if self._thread is not None:
            return
        if retry and self._pending is not None:
            message, context, key = self._pending
        else:
            message = self.message.toPlainText().strip()
            if not message:
                self.request_status.setText("Write a message first")
                return
            try:
                context = SelectedContext.create(self.context_type.currentData(), self.context_preview.toPlainText())
            except ValueError as exc:
                self.request_status.setText(str(exc))
                return
            key = f"client-{uuid.uuid4().hex}"
            self._pending = (message, context, key)
            self.workspace.record_event("ai.request", {"request_id": key, "context_type": context.kind, "context_sha256": context.sha256, "message_chars": len(message), "gateway_url": self.gateway_url.text().strip()}, outcome="sent")
            if self._ask_origin is not None:
                self.workspace.record_event("ai.context_sent", {"request_id": key, "action": self._ask_origin.action, "source_path": self._ask_origin.source_path, "file_hash": self._ask_origin.file_hash, "selection_hash": self._ask_origin.selection_hash, "code_chars": len(self._ask_origin.code)}, outcome="sent")
        self.send_button.setEnabled(False)
        self.retry_button.setVisible(False)
        self.request_status.setText("Waiting for gateway...")
        self._start_worker(_GatewayWorker("send", self._client(), message=message, context=context, key=key))

    def shutdown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(1000)
            self._thread = None
