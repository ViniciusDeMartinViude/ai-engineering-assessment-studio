"""Remote cluster controls. All network calls run off the Qt GUI thread."""
from __future__ import annotations
import os
from pathlib import Path
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QSpinBox, QComboBox, QMessageBox
)
from ..services.remote_cluster import RemoteClusterClient
from ..services.training import validate_subset

class Signals(QObject):
    result = Signal(object)
    error = Signal(str)

class Action(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = Signals()
    @Slot()
    def run(self):
        try:
            self.signals.result.emit(self.fn())
        except Exception as exc:
            self.signals.error.emit(str(exc))

class RemoteTrainingPanel(QWidget):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self.pool = QThreadPool.globalInstance()
        self.busy = False
        self.url = QLineEdit(os.getenv("AI_GPU_MANAGER_URL", ""))
        self.url.setPlaceholderText("http://MANAGER-LAN-IP:8000")
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Cluster API key (trusted operator only)")
        self.dataset = QLineEdit(os.getenv("AI_GPU_DATASET", "dates-v16"))
        self.dataset.setToolTip("Filled after uploading an M3 four-class subset; pre-staged IDs also work.")
        self.model = QComboBox()
        self.model.addItems(["yolo26n.pt", "yolo26s.pt"])
        self.epochs = QSpinBox(); self.epochs.setRange(1, 200); self.epochs.setValue(5)
        self.imgsz = QComboBox(); self.imgsz.addItems(["512", "640"])
        self.batch = QSpinBox(); self.batch.setRange(1, 8); self.batch.setValue(4)
        self.subset = QComboBox()
        self.subset.addItem("Select exported Dataset Studio subset", None)
        for candidate in sorted(workspace.resolve_inside("dataset").glob("subset-*")):
            try:
                validate_subset(workspace, candidate)
                self.subset.addItem(candidate.name, str(candidate))
            except Exception:
                pass
        self.upload_button = QPushButton("Upload selected subset")
        self.upload_button.clicked.connect(self.upload)
        layout = QVBoxLayout(self)
        note = QLabel("Direct GPU cluster training (trusted operator pilot). Upload a four-class Dataset Studio subset or use a pre-staged dataset ID.")
        note.setWordWrap(True); layout.addWidget(note)
        for caption, widget in (("Manager URL",self.url),("Cluster API key",self.token),("Dataset ID",self.dataset),
                                ("Model",self.model),("Epochs",self.epochs),("Image size",self.imgsz),("Batch",self.batch)):
            line=QHBoxLayout(); line.addWidget(QLabel(caption)); line.addWidget(widget, 1); layout.addLayout(line)
        selection = QHBoxLayout()
        selection.addWidget(self.subset, 1)
        selection.addWidget(self.upload_button)
        layout.addLayout(selection)
        buttons=QHBoxLayout()
        for caption, fn in (("Submit remote job",self.submit),("Refresh",self.refresh),
                            ("Cancel selected",self.cancel),("Download selected result",self.download)):
            button=QPushButton(caption);button.clicked.connect(fn);buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status=QLabel("Ready");layout.addWidget(self.status)
        self.table=QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Job ID", "Status", "GPU node", "Model"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

    def client(self):
        return RemoteClusterClient(self.url.text().strip(),self.token.text().strip(),
                                   self.workspace.session.candidate_id)

    def run_action(self, fn, callback):
        if self.busy: return
        self.busy=True
        task=Action(fn)
        task.signals.result.connect(self._finish)
        task.signals.error.connect(self._finish)
        task.signals.result.connect(callback)
        task.signals.error.connect(lambda err: self.status.setText("Error: "+err))
        self.pool.start(task)

    def _finish(self, *_):
        self.busy=False

    def upload(self):
        path = self.subset.currentData()
        if not path:
            self.status.setText("Choose an exported Dataset Studio subset")
            return
        try:
            validate_subset(self.workspace, Path(path))
            client = self.client()
        except Exception as exc:
            self.status.setText("Upload unavailable: " + str(exc))
            return
        self.status.setText("Uploading dataset...")
        self.run_action(lambda: client.upload_subset(Path(path)), self.upload_finished)

    def upload_finished(self, result):
        dataset_id = result.get("dataset_id")
        self.dataset.setText(dataset_id)
        self.status.setText("Dataset uploaded: " + dataset_id + ". Ready to train.")

    def submit(self):
        client=self.client()
        args=(self.model.currentText(),self.dataset.text().strip(),
              self.epochs.value(),int(self.imgsz.currentText()),self.batch.value())
        self.run_action(lambda:client.submit(*args),lambda job:(self.status.setText("Submitted: "+str(job.get("id", job.get("job_id", "?")))),self.refresh()))

    def refresh(self):
        client=self.client()
        self.run_action(client.jobs,self.display)

    def display(self, jobs):
        self.table.setRowCount(0)
        for job in jobs:
            row=self.table.rowCount();self.table.insertRow(row)
            for col,key in enumerate(("id","status","node","model")):
                self.table.setItem(row,col,QTableWidgetItem(str(job.get(key) or "")))
        self.status.setText(f"{len(jobs)} remote jobs")

    def selected_id(self):
        row=self.table.currentRow()
        return self.table.item(row,0).text() if row>=0 and self.table.item(row,0) else None

    def cancel(self):
        job=self.selected_id()
        if job:
            client=self.client()
            self.run_action(lambda:client.cancel(job),lambda _:(self.status.setText("Cancellation requested"),self.refresh()))

    def download(self):
        job=self.selected_id()
        if job:
            client=self.client()
            target=self.workspace.resolve_inside("exports","remote_training",job+".zip")
            self.run_action(lambda:client.download(job,target),
                            lambda path:self.status.setText("Saved result ZIP: "+str(path)))
