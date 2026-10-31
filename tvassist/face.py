"""Wil's talking head, floating on top of everything (top-right of the screen until you drag it).

Run with a Python that has PySide6 (the system one): face.py TOP.png JAW.png [HEIGHT] [POSITION.json].
Drag the head to move it; where you leave it is saved in POSITION.json. The two images
share one canvas: the head without its jaw, and the jaw. While the assistant speaks it writes
{"say": [loudness 0-1 per frame], "fps": 30} lines to stdin and the jaw drops in time with them.

It runs on XWayland (QT_QPA_PLATFORM=xcb) because Wayland doesn't let a window place itself.
Frameless, see-through around the cut-out (clicks there go to whatever is underneath), always on top.
"""
import json
import os
import signal
import sys
import threading
import time

from PySide6.QtCore import QObject, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPixmap, QRegion
from PySide6.QtWidgets import QApplication, QWidget

MARGIN = 24        # px from the screen's top and right edges
MAX_DROP = 0.07    # how far the jaw opens, as a share of the head's height
LATENCY = 0.08     # s between being told and the audio actually starting
MOUTH = QColor(40, 12, 14)


class Commands(QObject):
    said = Signal(list, float)
    gone = Signal()


class Head(QWidget):
    def __init__(self, top: QPixmap, jaw: QPixmap):
        super().__init__()
        self.top, self.jaw = top, jaw
        self.inside = QPixmap(jaw.size())  # the jaw's silhouette in mouth colour, for the gap
        self.inside.fill(Qt.GlobalColor.transparent)
        p = QPainter(self.inside)
        p.drawPixmap(0, 0, jaw)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(self.inside.rect(), MOUTH)
        p.end()
        self.max_drop = round(top.height() * MAX_DROP)
        self.drop = 0.0
        self.levels, self.fps, self.start = [], 30.0, 0.0
        self.resize(top.width(), top.height() + self.max_drop)
        self.setMask(self._shape())  # only the face itself takes the mouse
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.grab_at = None
        self.position_file = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def say(self, levels: list, fps: float) -> None:
        self.levels, self.fps, self.start = levels, fps, time.monotonic() + LATENCY

    def tick(self) -> None:
        i = int((time.monotonic() - self.start) * self.fps)
        target = self.levels[i] if 0 <= i < len(self.levels) else 0.0
        # open quickly, close a bit slower: reads as speech rather than flapping
        self.drop += (target - self.drop) * (0.6 if target > self.drop else 0.35)
        if abs(self.drop - target) < 0.01:
            self.drop = target
        self.update()

    def _shape(self) -> QRegion:
        """Everything the head covers with its mouth wide open."""
        full = QPixmap(self.size())
        full.fill(Qt.GlobalColor.transparent)
        p = QPainter(full)
        for k in range(self.max_drop + 1):
            p.drawPixmap(0, k, self.jaw)
        p.drawPixmap(0, 0, self.top)
        p.end()
        return QRegion(full.mask())

    # ---------- dragging ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.grab_at = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, e):
        if self.grab_at is not None:
            self.move(e.globalPosition().toPoint() - self.grab_at)

    def mouseReleaseEvent(self, e):
        if self.grab_at is not None:
            self.grab_at = None
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            if self.position_file:
                try:
                    with open(self.position_file, "w") as f:
                        json.dump({"x": self.x(), "y": self.y()}, f)
                except OSError:
                    pass

    def paintEvent(self, _):
        d = round(self.drop * self.max_drop)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        for k in range(d):  # the dark mouth the jaw sweeps open
            p.drawPixmap(0, k, self.inside)
        p.drawPixmap(0, d, self.jaw)
        p.drawPixmap(0, 0, self.top)
        p.end()


def main() -> None:
    top_path, jaw_path = sys.argv[1], sys.argv[2]
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 660
    position_file = sys.argv[4] if len(sys.argv) > 4 else None
    app = QApplication(sys.argv[:1])
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    pix = []
    for path in (top_path, jaw_path):
        p = QPixmap(path)
        if p.isNull():
            sys.exit(f"can't load {path}")
        pix.append(p.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation))

    w = Head(*pix)
    w.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                     | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
    w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    w.position_file = position_file
    screen = QGuiApplication.primaryScreen().availableGeometry()
    x, y = screen.right() - w.width() - MARGIN, screen.top() + MARGIN
    try:  # where it was last dragged to, if that's still on a screen
        with open(position_file) as f:
            saved = json.load(f)
        if any(s.geometry().intersects(w.rect().translated(saved["x"], saved["y"]))
               for s in QGuiApplication.screens()):
            x, y = saved["x"], saved["y"]
    except (OSError, TypeError, ValueError, KeyError):
        pass
    w.move(x, y)

    commands = Commands()
    commands.said.connect(w.say)
    commands.gone.connect(app.quit)

    def read_stdin():
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "say" in msg:
                commands.said.emit(msg["say"], float(msg.get("fps", 30)))
        commands.gone.emit()  # the assistant went away (quit on Qt's own thread)
    threading.Thread(target=read_stdin, daemon=True).start()

    w.setWindowOpacity(0.0)
    w.show()
    fade = QPropertyAnimation(w, b"windowOpacity")
    fade.setDuration(250)
    fade.setStartValue(0.0)
    fade.setEndValue(1.0)
    fade.start()
    code = app.exec()
    os._exit(code)  # not sys.exit: shutdown can hang on the thread still blocked reading stdin


if __name__ == "__main__":
    main()
