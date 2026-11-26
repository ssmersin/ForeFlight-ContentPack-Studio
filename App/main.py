from __future__ import annotations

import sys
import shutil
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QProgressBar,
    QPlainTextEdit, QVBoxLayout, QWidget,
)

from flip_parser import ChartInfo, DEFAULT_OUTPUT, inspect_pdf


class ExportWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(int, int, int)
    failed = Signal(str)

    def __init__(self, source: Path, destination: Path, charts: list[ChartInfo]):
        super().__init__()
        self.source = source
        self.destination = destination
        self.charts = charts

    @staticmethod
    def _ghostscript() -> Optional[str]:
        for name in ("gswin64c", "gswin32c", "gs"):
            found = shutil.which(name)
            if found:
                return found
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
        for root in filter(None, program_files):
            for candidate in Path(root).glob("gs/gs*/bin/gswin64c.exe"):
                return str(candidate)
        return None

    def run(self) -> None:
        try:
            ghostscript = self._ghostscript()
            if not ghostscript:
                raise RuntimeError(
                    "Ghostscript bulunamadı. PDF sayfalarının her birine tüm kaynak fontlarını kopyalayarak "
                    "çok büyük dosyalar üretmemek için Ghostscript gereklidir. 64-bit Ghostscript'i kurup "
                    "uygulamayı yeniden başlatın: https://www.ghostscript.com/releases/"
                )
            self.destination.mkdir(parents=True, exist_ok=True)
            exported = skipped = 0
            for position, chart in enumerate(self.charts, start=1):
                target = self.destination / chart.filename
                if target.exists():
                    skipped += 1
                    self.progress.emit(position, len(self.charts), f"Atlandı (dosya zaten var): {chart.filename}")
                    continue
                command = [
                    ghostscript,
                    "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE",
                    "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.7",
                    "-dPDFSETTINGS=/prepress", "-dEmbedAllFonts=true",
                    "-dSubsetFonts=true", "-dCompressFonts=true",
                    f"-dFirstPage={chart.pdf_page}", f"-dLastPage={chart.pdf_page}",
                    f"-sOutputFile={target}", str(self.source),
                ]
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if result.returncode != 0 or not target.exists() or target.stat().st_size == 0:
                    target.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"PDF sayfası dışarı aktarılamadı (PDF sayfa {chart.pdf_page}).\n"
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
                exported += 1
                self.progress.emit(position, len(self.charts), f"Aktarıldı: {chart.filename}")
            self.finished_ok.emit(exported, skipped, len(self.charts))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class AnalysisWorker(QThread):
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, source: Path, effective_date: str, document_type: str):
        super().__init__()
        self.source = source
        self.effective_date = effective_date
        self.document_type = document_type

    def run(self) -> None:
        try:
            charts, unresolved = inspect_pdf(self.source, self.effective_date, self.document_type)
            self.completed.emit(charts, unresolved)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ForeFlight Content Pack Studio")
        self.resize(820, 620)
        self.source_path: Optional[Path] = None
        self.charts: list[ChartInfo] = []
        self.unresolved: list[tuple[int, str]] = []
        self.worker: Optional[ExportWorker] = None
        self.analysis_worker: Optional[AnalysisWorker] = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        heading = QLabel("FLIP PDF – İlk Ayrıştırma")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Kaynak PDF içindeki meydan ve prosedür chart'larını tek sayfalık PDF dosyalarına ayırır."))

        form = QFormLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_edit, 1)
        source_btn = QPushButton("PDF Seç…")
        source_btn.clicked.connect(self.choose_source)
        source_row.addWidget(source_btn)
        form.addRow("Kaynak PDF", source_row)

        self.date_edit = QLineEdit("MAR2027")
        self.type_edit = QLineEdit("FLIP")
        form.addRow("Yürürlük tarihi", self.date_edit)
        form.addRow("Doküman türü", self.type_edit)

        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_btn = QPushButton("Klasör Seç…")
        output_btn.clicked.connect(self.choose_output)
        output_row.addWidget(output_btn)
        form.addRow("Hedef klasör", output_row)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.inspect_btn = QPushButton("Sayfaları Analiz Et")
        self.inspect_btn.clicked.connect(self.inspect)
        self.export_btn = QPushButton("Initial Klasörüne Aktar")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export)
        actions.addWidget(self.inspect_btn)
        actions.addWidget(self.export_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.summary = QLabel("Henüz PDF seçilmedi.")
        self.summary.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.summary)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        self.statusBar().showMessage("Hazır")

    def choose_source(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "FLIP PDF seç", "", "PDF dosyaları (*.pdf)")
        if filename:
            self.source_path = Path(filename)
            self.source_edit.setText(str(self.source_path))
            self.charts = []
            self.export_btn.setEnabled(False)
            self.summary.setText("PDF seçildi. Analiz başlatılabilir.")
            self.log.clear()

    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Hedef klasörü seç", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)

    def inspect(self) -> None:
        if not self.source_path or not self.source_path.is_file():
            QMessageBox.warning(self, "Kaynak gerekli", "Lütfen önce bir PDF dosyası seçin.")
            return
        if not self.date_edit.text().strip() or not self.type_edit.text().strip():
            QMessageBox.warning(self, "Eksik bilgi", "Doküman türü ve yürürlük tarihi boş bırakılamaz.")
            return
        self.inspect_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.summary.setText("PDF sayfaları analiz ediliyor…")
        self.log.clear()
        self.statusBar().showMessage("PDF analiz ediliyor…")
        self.analysis_worker = AnalysisWorker(self.source_path, self.date_edit.text(), self.type_edit.text())
        self.analysis_worker.completed.connect(self.on_analysis_completed)
        self.analysis_worker.failed.connect(self.on_analysis_failed)
        self.analysis_worker.start()

    def on_analysis_completed(self, charts: list[ChartInfo], unresolved: list[tuple[int, str]]) -> None:
        self.charts = charts
        self.unresolved = unresolved
        self.summary.setText(f"Bulunan chart: {len(self.charts)} | İncelenmesi gereken sayfa: {len(self.unresolved)}")
        for chart in self.charts:
            self.log.appendPlainText(f"PDF {chart.pdf_page:>3}  →  {chart.filename}")
        if self.unresolved:
            self.log.appendPlainText("\n--- Otomatik eşleştirilemeyen olası chart sayfaları ---")
            for page_no, reason in self.unresolved:
                self.log.appendPlainText(f"PDF {page_no}: {reason}")
        self.export_btn.setEnabled(bool(self.charts))
        self.inspect_btn.setEnabled(True)
        self.statusBar().showMessage("Analiz tamamlandı")

    def on_analysis_failed(self, message: str) -> None:
        self.inspect_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.summary.setText("Analiz tamamlanamadı.")
        QMessageBox.critical(self, "PDF analiz hatası", message)
        self.statusBar().showMessage("PDF analizi başarısız")

    def export(self) -> None:
        if not self.charts:
            return
        destination = Path(self.output_edit.text()).expanduser()
        answer = QMessageBox.question(
            self, "Aktarımı onayla",
            f"{len(self.charts)} chart sayfası şu klasöre aktarılacak:\n\n{destination}\n\nMevcut dosyalar üzerine yazılmayacak. Devam edilsin mi?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.inspect_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress.setValue(0)
        self.worker = ExportWorker(self.source_path, destination, self.charts)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(current)
        self.log.appendPlainText(message)
        self.statusBar().showMessage(f"Aktarım: {current}/{total}")

    def on_finished(self, exported: int, skipped: int, total: int) -> None:
        self.inspect_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.summary.setText(f"Tamamlandı | Aktarılan: {exported} | Atlanan: {skipped} | Toplam: {total}")
        self.statusBar().showMessage("Aktarım tamamlandı")
        QMessageBox.information(self, "Aktarım tamamlandı", f"Aktarılan: {exported}\nZaten mevcut olduğu için atlanan: {skipped}\nToplam chart: {total}\n\nHedef: {self.output_edit.text()}")

    def on_failed(self, message: str) -> None:
        self.inspect_btn.setEnabled(True)
        self.export_btn.setEnabled(bool(self.charts))
        QMessageBox.critical(self, "Aktarım hatası", message)
        self.statusBar().showMessage("Aktarım başarısız")


    def closeEvent(self, event) -> None:
        if self.analysis_worker and self.analysis_worker.isRunning():
            QMessageBox.information(self, "İşlem sürüyor", "PDF analizi tamamlana kadar uygulamayı kapatmayın.")
            event.ignore()
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "İİşlem sürüyor", "PDF aktarımı tamamlana kadar uygulamayı kapatmayın.")
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("windowsvista")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
