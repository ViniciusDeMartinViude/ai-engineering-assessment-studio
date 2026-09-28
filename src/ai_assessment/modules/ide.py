from __future__ import annotations

import difflib
from pathlib import Path

from PySide6.QtCore import QModelIndex, QObject, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QTextCursor, QTextFormat
from PySide6.QtWidgets import (
    QFileDialog,
    QFileSystemModel,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolButton,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..core.workspace import CandidateWorkspace
from ..services.ai_context import AIContextBuffer
from ..services.ai_ide_transfer import AIIdeCoordinator, AIAskRequest, IDEHandoff, sha256_text
from ..services.execution import (
    ExecutionError,
    ExecutionResult,
    ExecutionService,
    TerminalService,
    parse_arguments,
)


class LineNumberArea(QWidget):
    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self.editor.line_number_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.editor.paint_line_numbers(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("codeEditor")
        self.setFont(QFont("Consolas", 10))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.line_numbers = LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_width)
        self.updateRequest.connect(self._update_line_numbers)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_line_number_width(0)
        self._highlight_current_line()
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))

    def line_number_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_number_width(self, _count: int) -> None:
        self.setViewportMargins(self.line_number_width(), 0, 0, 0)

    def _update_line_numbers(self, rect: QRect, dy: int) -> None:
        if dy:
            self.line_numbers.scroll(0, dy)
        else:
            self.line_numbers.update(0, rect.y(), self.line_numbers.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_width(0)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        contents = self.contentsRect()
        self.line_numbers.setGeometry(QRect(contents.left(), contents.top(), self.line_number_width(), contents.height()))

    def paint_line_numbers(self, event) -> None:
        painter = QPainter(self.line_numbers)
        painter.fillRect(event.rect(), QColor("#eef3f0"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor("#77837d"))
                painter.drawText(0, top, self.line_numbers.width() - 6, self.fontMetrics().height(), Qt.AlignmentFlag.AlignRight, str(block_number + 1))
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def _highlight_current_line(self) -> None:
        selections = []
        if not self.isReadOnly():
            selection = QTextFormat.FullWidthSelection
            extra = QTextEdit.ExtraSelection()
            extra.format.setBackground(QColor("#f3f8f5"))
            extra.format.setProperty(selection, True)
            extra.cursor = self.textCursor()
            extra.cursor.clearSelection()
            selections.append(extra)
        self.setExtraSelections(selections)


class _Signals(QObject):
    output = Signal(str, str)
    completed = Signal(object)
    terminal_output = Signal(str, str)
    terminal_exited = Signal()


class IDEPage(QWidget):
    def __init__(self, workspace: CandidateWorkspace, context_buffer: AIContextBuffer | None = None, coordinator: AIIdeCoordinator | None = None) -> None:
        super().__init__()
        self.workspace = workspace
        self.context_buffer = context_buffer or AIContextBuffer()
        self.coordinator = coordinator
        if self.coordinator is not None:
            self.coordinator.handoff_requested.connect(self._receive_handoff)
        self.execution = ExecutionService(workspace)
        self.terminal = TerminalService(workspace)
        self.signals = _Signals()
        self.signals.output.connect(self._append_execution_output)
        self.signals.completed.connect(self._execution_completed)
        self.signals.terminal_output.connect(self._append_terminal_output)
        self.signals.terminal_exited.connect(self._terminal_exited)
        self._max_output_chars = 100_000
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(10)
        heading = QHBoxLayout()
        title = QLabel("IDE")
        title.setObjectName("pageTitle")
        heading.addWidget(title)
        heading.addStretch(1)
        root.addLayout(heading)
        summary = QLabel(
            "Edit and run candidate-owned Python files from workspace/src. Runs are local process executions, not an OS security sandbox."
        )
        summary.setObjectName("summary")
        summary.setWordWrap(True)
        root.addWidget(summary)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_tree())
        split.addWidget(self._build_workbench())
        split.setSizes([260, 860])
        root.addWidget(split, 1)

    def _build_tree(self) -> QWidget:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.addWidget(QLabel("Candidate workspace"))
        self.file_model = QFileSystemModel(self)
        self.file_model.setRootPath(str(self.workspace.root))
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(str(self.workspace.root)))
        self.file_tree.setHeaderHidden(False)
        self.file_tree.doubleClicked.connect(self._open_tree_item)
        layout.addWidget(self.file_tree, 1)
        new_button = QPushButton("New Python file")
        new_button.clicked.connect(self._new_file)
        layout.addWidget(new_button)
        return frame

    def _build_workbench(self) -> QWidget:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 0, 0, 0)

        editor_panel = QFrame()
        editor_panel.setObjectName("editorPanel")
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.setContentsMargins(10, 10, 10, 10)
        editor_header = QHBoxLayout()
        editor_title = QLabel("Python editor")
        editor_title.setObjectName("sectionHeading")
        editor_header.addWidget(editor_title)
        editor_header.addWidget(QLabel("Candidate files under src/"))
        editor_header.addStretch(1)
        editor_layout.addLayout(editor_header)

        toolbar = QHBoxLayout()
        for label, handler in (("Save", self._save_current), ("Save As", self._save_as), ("Run Python", self._run), ("Stop", self._stop)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            toolbar.addWidget(button)
        attach = QPushButton("Attach selection")
        attach.setToolTip("Make the selected editor text available to AI Assistant for deliberate attachment")
        attach.clicked.connect(self._attach_selection)
        toolbar.addWidget(attach)
        ask_button = QToolButton()
        ask_button.setText("Ask AI")
        ask_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        ask_menu = QMenu(ask_button)
        for action_name in ("Explain", "Fix", "Improve", "Generate code"):
            action = ask_menu.addAction(action_name)
            action.triggered.connect(lambda _checked=False, name=action_name: self._ask_ai(name))
        ask_button.setMenu(ask_menu)
        toolbar.addWidget(ask_button)
        toolbar.addWidget(QLabel("Arguments:"))
        self.arguments = QLineEdit()
        self.arguments.setPlaceholderText("optional arguments, parsed into an argument array")
        toolbar.addWidget(self.arguments, 1)
        editor_layout.addLayout(toolbar)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("editorTabs")
        self.tabs.setTabsClosable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setMinimumHeight(250)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._current_changed)
        editor_layout.addWidget(self.tabs, 1)
        layout.addWidget(editor_panel, 3)

        run_panel = QFrame()
        run_panel.setObjectName("runPanel")
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(10, 8, 10, 10)
        run_header = QHBoxLayout()
        run_title = QLabel("Run output")
        run_title.setObjectName("sectionHeading")
        run_header.addWidget(run_title)
        self.execution_status = QLabel("No Python run active")
        run_header.addWidget(self.execution_status)
        run_header.addStretch(1)
        run_layout.addLayout(run_header)
        self.execution_output = QPlainTextEdit()
        self.execution_output.setObjectName("outputConsole")
        self.execution_output.setFont(QFont("Consolas", 9))
        self.execution_output.setReadOnly(True)
        self.execution_output.setMaximumBlockCount(2500)
        self.execution_output.setPlaceholderText("Python stdout and stderr will appear here.")
        run_layout.addWidget(self.execution_output)
        layout.addWidget(run_panel, 2)

        terminal_panel = QFrame()
        terminal_panel.setObjectName("terminalPanel")
        terminal_layout = QVBoxLayout(terminal_panel)
        terminal_layout.setContentsMargins(10, 8, 10, 10)
        terminal_title = QHBoxLayout()
        terminal_heading = QLabel("Terminal")
        terminal_heading.setObjectName("sectionHeading")
        terminal_title.addWidget(terminal_heading)
        terminal_title.addWidget(QLabel("pipe-based process"))
        self.terminal_status = QLabel("Stopped")
        terminal_title.addWidget(self.terminal_status)
        terminal_title.addStretch(1)
        for label, handler in (("Start", self._start_terminal), ("Restart", self._restart_terminal), ("Stop", self._stop_terminal)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            terminal_title.addWidget(button)
        terminal_layout.addLayout(terminal_title)
        limitation = QLabel("This terminal uses pipes: interactive console control sequences and secure prompt behavior may be limited. Input is never recorded.")
        limitation.setWordWrap(True)
        terminal_layout.addWidget(limitation)
        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setObjectName("outputConsole")
        self.terminal_output.setFont(QFont("Consolas", 9))
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setMaximumBlockCount(2500)
        terminal_layout.addWidget(self.terminal_output)
        terminal_input = QHBoxLayout()
        self.terminal_command = QLineEdit()
        self.terminal_command.setPlaceholderText("command")
        self.terminal_command.returnPressed.connect(self._send_terminal)
        send = QPushButton("Send")
        send.clicked.connect(self._send_terminal)
        terminal_input.addWidget(self.terminal_command, 1)
        terminal_input.addWidget(send)
        terminal_layout.addLayout(terminal_input)
        layout.addWidget(terminal_panel, 2)
        frame.setStyleSheet(
            """
            #editorPanel, #runPanel, #terminalPanel {
                background: #ffffff;
                border: 1px solid #b7c8bf;
                border-radius: 6px;
            }
            #sectionHeading {
                color: #173c2b;
                font-size: 15px;
                font-weight: 700;
            }
            #codeEditor, #outputConsole {
                background: #fbfdfc;
                border: 1px solid #819b8e;
                border-radius: 4px;
                selection-background-color: #cfe5d9;
            }
            #editorTabs::pane {
                border: 1px solid #819b8e;
                border-top: 0px;
                background: #fbfdfc;
            }
            #editorTabs::tab {
                background: #e7efeb;
                border: 1px solid #b7c8bf;
                border-bottom: 0px;
                padding: 6px 12px;
                min-width: 110px;
            }
            #editorTabs::tab:selected {
                background: #fbfdfc;
                color: #173c2b;
                font-weight: 700;
            }
            """
        )
        return frame

    def _new_file(self) -> None:
        editor = CodeEditor()
        editor.setProperty("path", None)
        editor.document().setModified(True)
        editor.document().modificationChanged.connect(lambda _changed, e=editor: self._refresh_tab(e))
        index = self.tabs.addTab(editor, "* untitled.py")
        self.tabs.setCurrentIndex(index)
        editor.setFocus()

    def _open_tree_item(self, index: QModelIndex) -> None:
        path = Path(self.file_model.filePath(index))
        if not path.is_file():
            return
        try:
            _, text = self.execution.load_file(path)
        except (OSError, ExecutionError) as exc:
            self._show_error(str(exc))
            return
        for tab_index in range(self.tabs.count()):
            if self.tabs.widget(tab_index).property("path") == str(path):
                self.tabs.setCurrentIndex(tab_index)
                return
        editor = CodeEditor()
        editor.setPlainText(text)
        editor.document().setModified(False)
        editor.setProperty("path", str(path))
        readonly = path.relative_to(self.workspace.root).parts[0].lower() == "dataset"
        editor.setReadOnly(readonly)
        index = self.tabs.addTab(editor, path.name)
        self.tabs.setCurrentIndex(index)
        editor.document().modificationChanged.connect(lambda _changed, e=editor: self._refresh_tab(e))

    def _current_editor(self) -> CodeEditor | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, CodeEditor) else None

    def _current_changed(self, _index: int) -> None:
        editor = self._current_editor()
        if editor is not None:
            self._refresh_tab(editor)

    def _refresh_tab(self, editor: CodeEditor) -> None:
        index = self.tabs.indexOf(editor)
        path = editor.property("path")
        name = Path(path).name if path else "untitled.py"
        self.tabs.setTabText(index, ("* " if editor.document().isModified() else "") + name)

    def _save_current(self) -> bool:
        editor = self._current_editor()
        if editor is None:
            return False
        path = editor.property("path")
        if not path:
            return self._save_as()
        try:
            saved, _ = self.execution.save_file(path, editor.toPlainText())
        except (OSError, ExecutionError) as exc:
            self._show_error(str(exc))
            return False
        editor.setProperty("path", str(saved))
        editor.document().setModified(False)
        self.file_model.setRootPath(str(self.workspace.root))
        self._refresh_tab(editor)
        return True

    def _save_as(self) -> bool:
        editor = self._current_editor()
        if editor is None:
            return False
        default = str(self.workspace.root / "src" / "untitled.py")
        filename, _ = QFileDialog.getSaveFileName(self, "Save Python file", default, "Python files (*.py);;All files (*.*)")
        if not filename:
            return False
        editor.setProperty("path", filename)
        return self._save_current()

    def _run(self) -> None:
        editor = self._current_editor()
        if editor is None:
            self._show_error("Open a Python file before running it.")
            return
        if editor.isReadOnly():
            self._show_error("Dataset files and other read-only files cannot be run from the editor.")
            return
        if editor.document().isModified() and not self._save_current():
            return
        path = editor.property("path")
        try:
            args = parse_arguments(self.arguments.text())
            run_id = self.execution.run_script(path, args, on_output=lambda s, t: self.signals.output.emit(s, t), on_complete=lambda r: self.signals.completed.emit(r))
        except (ExecutionError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.execution_status.setText(f"Running {run_id}")

    def _attach_selection(self) -> None:
        editor = self._current_editor()
        if editor is None:
            self.execution_status.setText("Open an editor tab first")
            return
        text = editor.textCursor().selectedText()
        if not text:
            self.execution_status.setText("Select text before attaching it")
            return
        self.context_buffer.set("selected_text", text, "IDE selection")
        self.execution_status.setText(f"Attached {len(text)} characters for AI Assistant")

    def _ask_ai(self, action: str) -> None:
        editor = self._current_editor()
        if editor is None:
            self.execution_status.setText("Open an editor tab before asking AI")
            return
        whole_file = editor.toPlainText()
        selected = editor.textCursor().selectedText().replace("\u2029", "\n")
        has_selection = bool(selected)
        if not has_selection:
            answer = QMessageBox.question(
                self,
                "Send current file to AI?",
                "Nothing is selected. Do you explicitly want to send the current file contents for this request?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.execution_status.setText("AI request cancelled; no code was sent")
                return
            code = whole_file
        else:
            code = selected
        path = editor.property("path")
        source_path = None
        if path:
            try:
                source_path = self.execution.relative(self.execution.resolve_path(path))
            except ExecutionError:
                source_path = None
        questions = {
            "Explain": "Explain this code, including its assumptions and important edge cases.",
            "Fix": "Review this code for bugs and propose a corrected version.",
            "Improve": "Suggest focused improvements while preserving the intended behavior.",
            "Generate code": "Generate code that completes the requested task using this context.",
        }
        request = AIAskRequest(
            action=action,
            question=questions[action],
            source_path=source_path,
            code=code,
            file_hash=sha256_text(whole_file),
            selection_hash=sha256_text(code),
            has_selection=has_selection,
        )
        if self.coordinator is None:
            self.execution_status.setText("AI integration is unavailable")
            return
        self.coordinator.request_ai(request)
        self.execution_status.setText("AI preview ready; review it in AI Assistant before sending")

    def _receive_handoff(self, handoff: IDEHandoff) -> None:
        editor = self._current_editor()
        stale = self._handoff_stale(handoff)
        action = self._choose_handoff_action(handoff, editor, stale)
        if action == "cancel":
            self.execution_status.setText("IDE handoff cancelled")
            return
        if action == "replace" and editor is not None and not self._confirm_replacement(editor, handoff.block.code):
            self.execution_status.setText("Replacement cancelled")
            return
        if self._apply_handoff(handoff, action):
            self.execution_status.setText(f"{handoff.block.block_id} opened in IDE as an unsaved draft")

    def _choose_handoff_action(self, handoff: IDEHandoff, editor: CodeEditor | None, stale: bool) -> str:
        dialog = QDialog(self)
        dialog.setWindowTitle("Send generated code to IDE")
        layout = QVBoxLayout(dialog)
        label = QLabel(f"Preview for {handoff.block.block_id} ({handoff.block.language})")
        layout.addWidget(label)
        if stale:
            warning = QLabel("The original file, selection, or current tab changed. Choose a destination deliberately; no dirty tab will be overwritten silently.")
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #8a4b08; font-weight: 700;")
            layout.addWidget(warning)
        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setPlainText(handoff.block.code)
        preview.setMinimumSize(600, 260)
        layout.addWidget(preview)
        buttons = QHBoxLayout()
        choices = (("New file", "new_file"), ("Insert at cursor", "insert"), ("Replace selection", "replace"))
        for title, action in choices:
            button = QPushButton(title)
            button.setEnabled(action == "new_file" or editor is not None)
            if action == "replace" and (editor is None or not editor.textCursor().hasSelection()):
                button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, value=action: dialog.done({"new_file": 1, "insert": 2, "replace": 3}[value]))
            buttons.addWidget(button)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dialog.reject)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        result = dialog.exec()
        return {1: "new_file", 2: "insert", 3: "replace"}.get(result, "cancel")

    def _confirm_replacement(self, editor: CodeEditor, code: str) -> bool:
        old = editor.textCursor().selectedText().replace("\u2029", "\n")
        diff = "".join(difflib.unified_diff(old.splitlines(True), code.splitlines(True), fromfile="current selection", tofile="generated code"))
        dialog = QDialog(self)
        dialog.setWindowTitle("Preview replacement")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Review this diff before replacing the selected text."))
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(diff or code)
        view.setMinimumSize(600, 260)
        layout.addWidget(view)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(dialog.accept)
        box.rejected.connect(dialog.reject)
        layout.addWidget(box)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def _handoff_stale(self, handoff: IDEHandoff) -> bool:
        if not any((handoff.source_path, handoff.source_file_hash, handoff.source_selection_hash)):
            return False
        editor = self._current_editor()
        if editor is None:
            return True
        current_path = editor.property("path")
        try:
            current_relative = self.execution.relative(self.execution.resolve_path(current_path)) if current_path else None
        except ExecutionError:
            return True
        if handoff.source_path != current_relative:
            return True
        if handoff.source_file_hash and sha256_text(editor.toPlainText()) != handoff.source_file_hash:
            return True
        selected = editor.textCursor().selectedText().replace("\u2029", "\n")
        if handoff.source_selection_hash and sha256_text(selected) != handoff.source_selection_hash:
            return True
        return editor.document().isModified()

    def _apply_handoff(self, handoff: IDEHandoff, action: str) -> bool:
        if action == "cancel":
            return False
        editor = self._current_editor()
        if action == "new_file":
            editor = CodeEditor()
            editor.setProperty("path", None)
            editor.setPlainText(handoff.block.code)
            editor.document().setModified(True)
            editor.document().modificationChanged.connect(lambda _changed, e=editor: self._refresh_tab(e))
            index = self.tabs.addTab(editor, f"* generated_{handoff.block.block_id}.py")
            self.tabs.setCurrentIndex(index)
            editor.setFocus()
        elif editor is None:
            return False
        elif action == "insert":
            cursor = editor.textCursor()
            cursor.clearSelection()
            editor.setTextCursor(cursor)
            cursor.insertText(handoff.block.code)
            editor.setTextCursor(cursor)
        elif action == "replace":
            cursor = editor.textCursor()
            if not cursor.hasSelection():
                return False
            cursor.insertText(handoff.block.code)
            editor.setTextCursor(cursor)
        else:
            return False
        target_path = None
        if editor.property("path"):
            try:
                target_path = self.execution.relative(self.execution.resolve_path(editor.property("path")))
            except ExecutionError:
                target_path = None
        self.workspace.record_event("ai.ide_handoff", {"response_id": handoff.response_id, "block_id": handoff.block.block_id, "code_sha256": handoff.block.code_hash, "action": action, "target_path": target_path}, outcome="accepted")
        return True

    def _stop(self) -> None:
        if not self.execution.stop():
            self.execution_status.setText("No Python run active")

    def _append_execution_output(self, stream: str, text: str) -> None:
        self._append_bounded(self.execution_output, stream, text)

    def _execution_completed(self, result: ExecutionResult) -> None:
        message = f"{result.status}: exit {result.exit_code} in {result.duration_ms} ms ({result.path})"
        self.execution_status.setText(message)

    def _start_terminal(self) -> None:
        try:
            self.terminal.start(on_output=lambda s, t: self.signals.terminal_output.emit(s, t), on_exit=lambda: self.signals.terminal_exited.emit())
        except ExecutionError as exc:
            self._show_error(str(exc))
            return
        self.terminal_status.setText("Running")

    def _restart_terminal(self) -> None:
        try:
            self.terminal.restart(on_output=lambda s, t: self.signals.terminal_output.emit(s, t), on_exit=lambda: self.signals.terminal_exited.emit())
        except ExecutionError as exc:
            self._show_error(str(exc))
            return
        self.terminal_status.setText("Running")

    def _stop_terminal(self) -> None:
        self.terminal.stop()
        self.terminal_status.setText("Stopped")

    def _send_terminal(self) -> None:
        command = self.terminal_command.text()
        if not command:
            return
        try:
            self.terminal.write(command)
        except ExecutionError as exc:
            self._show_error(str(exc))
            return
        self.terminal_command.clear()

    def _append_terminal_output(self, stream: str, text: str) -> None:
        self._append_bounded(self.terminal_output, stream, text)

    def _append_bounded(self, output: QPlainTextEdit, stream: str, text: str) -> None:
        output.appendPlainText(f"[{stream}] {text.rstrip()}")
        content = output.toPlainText()
        if len(content) > self._max_output_chars:
            output.setPlainText(content[-self._max_output_chars :])
            output.moveCursor(QTextCursor.MoveOperation.End)

    def _terminal_exited(self) -> None:
        self.terminal_status.setText("Stopped")

    def _close_tab(self, index: int) -> None:
        editor = self.tabs.widget(index)
        if isinstance(editor, CodeEditor) and editor.document().isModified():
            answer = QMessageBox.question(self, "Unsaved changes", "Save changes before closing this tab?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Save:
                self.tabs.setCurrentIndex(index)
                if not self._save_current():
                    return
        self.tabs.removeTab(index)
        editor.deleteLater()

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "IDE", message)

    def shutdown(self) -> None:
        for index in range(self.tabs.count() - 1, -1, -1):
            editor = self.tabs.widget(index)
            if isinstance(editor, CodeEditor) and editor.document().isModified():
                self.tabs.setCurrentIndex(index)
                answer = QMessageBox.question(self, "Unsaved changes", "Save changes before closing the application?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard)
                if answer == QMessageBox.StandardButton.Save:
                    self._save_current()
        self.execution.shutdown()
        self.terminal.shutdown()
