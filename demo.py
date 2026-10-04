import copy
import multiprocessing as mp
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"

import neat
import pygame
from PySide6 import QtCore, QtGui, QtWidgets, QtSvgWidgets
from PySide6.QtCore import Qt, Signal, Slot

from Simulation import Simulation
from SimulationForParallel import SimulationForParallel
import visualize


def eval_genome_parallel(genome, config):
    genome.fitness = 0.0
    sim = SimulationForParallel()
    sim.make_walker()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    for _ in range(1500):
        if sim.walker.is_dead():
            break
        inputs = sim.walker.info().as_array()
        outputs = net.activate(inputs)
        sim.update(outputs)

    return sim.walker.fitness()


class ScreenWidget(QtWidgets.QWidget):
    def __init__(self, width=800, height=600, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB888)
        self.clear_screen()

    def clear_screen(self):
        self.image.fill(Qt.GlobalColor.black)
        self.update()

    @Slot(bytes, int, int)
    def update_frame(self, frame_bytes, w, h):
        bytes_per_line = w * 3
        img = QtGui.QImage(
            frame_bytes,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        self.image = img
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)


class InteractiveImageViewer(QtWidgets.QDialog):
    def __init__(self, svg_path=None, fallback_pixmap=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Podgląd struktury sieci (Kółko myszy = Zoom, Przeciąganie = Przesuwanie, Spacja = Reset, ESC = Wyjście)")
        self.resize(1200, 800)
        self.setWindowState(Qt.WindowState.WindowMaximized)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #1e1e1e; border: none;")

        self.view.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        self.view.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout.addWidget(self.view)

        self.target_item = None

        if svg_path and os.path.exists(svg_path) and svg_path.lower().endswith(".svg"):
            self.svg_item = QtSvgWidgets.QGraphicsSvgItem(svg_path)
            self.scene.addItem(self.svg_item)
            self.target_item = self.svg_item
        elif fallback_pixmap and not fallback_pixmap.isNull():
            self.target_item = self.scene.addPixmap(fallback_pixmap)

        QtCore.QTimer.singleShot(50, self.fit_to_view)

    def fit_to_view(self):
        if self.target_item:
            self.view.fitInView(self.target_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.view.scale(zoom_factor, zoom_factor)
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.fit_to_view()
        else:
            super().keyPressEvent(event)


class ClickableImageLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.current_file_path = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_custom_image(self, file_path):
        self.current_file_path = file_path
        pixmap = QtGui.QPixmap(file_path)
        if not pixmap.isNull():
            self.current_pixmap = pixmap
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap and not self.current_pixmap.isNull():
            scaled = self.current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_file_path:
                viewer = InteractiveImageViewer(
                    svg_path=self.current_file_path,
                    fallback_pixmap=self.current_pixmap,
                    parent=self,
                )
                viewer.exec()
        super().mousePressEvent(event)


class TrainingWorker(QtCore.QThread):
    frame_ready = Signal(bytes, int, int)
    epoch_recorded = Signal(int, float, object)
    status_changed = Signal(str)
    training_done = Signal(object, object)

    def __init__(self, config_path, neat_params, total_epochs, render_enabled=False):
        super().__init__()
        self.config_path = config_path
        self.neat_params = neat_params
        self.total_epochs = total_epochs
        self.render_enabled = render_enabled
        self.is_running = True

    def run(self):
        self.status_changed.emit("Inicjalizacja NEAT...")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path,
        )

        config.pop_size = self.neat_params['pop_size']

        gc = config.genome_config
        gc.initial_connection = self.neat_params['initial_connection']
        gc.conn_add_prob = self.neat_params['conn_add_prob']
        gc.conn_delete_prob = self.neat_params['conn_delete_prob']
        gc.bias_mutate_power = self.neat_params['bias_mutate_power']
        gc.bias_mutate_rate = self.neat_params['bias_mutate_rate']
        gc.weight_mutate_power = self.neat_params['weight_mutate_power']
        gc.enabled_mutate_rate = self.neat_params['enabled_mutate_rate']

        config.species_set_config.compatibility_threshold = self.neat_params['compatibility_threshold']
        config.reproduction_config.survival_threshold = self.neat_params['survival_threshold']

        population = neat.Population(config)

        sim = None
        if self.render_enabled:
            sim = Simulation()

        for epoch in range(self.total_epochs):
            if not self.is_running:
                break

            current_epoch = epoch + 1
            self.status_changed.emit(f"Epoka {current_epoch}/{self.total_epochs}")

            if self.render_enabled and sim is not None:
                def eval_genomes_gui(genomes, cfg):
                    sim.reset()
                    sim.make_walkers(len(genomes))
                    nets = [
                        neat.nn.FeedForwardNetwork.create(g, cfg)
                        for _, g in genomes
                    ]

                    for _ in range(1500):
                        if not self.is_running:
                            break

                        inputs = sim.infos_array()
                        outputs = [
                            net.activate(inputs[i]) for i, net in enumerate(nets)
                        ]

                        sim.handle_events()
                        sim.update(outputs)
                        sim.draw()

                        frame_bytes = pygame.image.tobytes(sim.screen, "RGB")
                        w, h = sim.screen.get_size()
                        self.frame_ready.emit(frame_bytes, w, h)

                    for i, (_, genome) in enumerate(genomes):
                        genome.fitness = sim.walkers[i].fitness()

                population.run(eval_genomes_gui, 1)
            else:
                pe = neat.ParallelEvaluator(
                    mp.cpu_count(), eval_genome_parallel
                )
                population.run(pe.evaluate, 1)

            best_genome_this_epoch = max(
                population.population.values(),
                key=lambda g: g.fitness if g.fitness is not None else -float('inf'),
            )
            best_fit = (
                best_genome_this_epoch.fitness
                if best_genome_this_epoch.fitness is not None
                else 0.0
            )

            genome_snapshot = copy.deepcopy(best_genome_this_epoch)
            self.epoch_recorded.emit(current_epoch, best_fit, genome_snapshot)

        self.status_changed.emit("Trening zakończony / wstrzymany")
        self.training_done.emit(population.best_genome, config)

    def stop(self):
        self.is_running = False


class PlaybackWorker(QtCore.QThread):
    frame_ready = Signal(bytes, int, int)
    playback_done = Signal()

    def __init__(self, genome_to_play, config):
        super().__init__()
        self.genome = genome_to_play
        self.config = config
        self.is_running = True

    def run(self):
        sim = Simulation()
        sim.reset()
        sim.make_walkers(1)
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        for _ in range(1500):
            if not self.is_running:
                break

            inputs = sim.walkers[0].info().as_array()
            outputs = net.activate(inputs)

            sim.handle_events()
            sim.update([outputs])
            sim.draw()

            frame_bytes = pygame.image.tobytes(sim.screen, "RGB")
            w, h = sim.screen.get_size()
            self.frame_ready.emit(frame_bytes, w, h)

            if sim.walkers[0].info().headAltitude < 0.4:
                break

        self.playback_done.emit()

    def stop(self):
        self.is_running = False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Walker NEAT - Sterowanie i Ewaluacja")
        self.resize(1340, 880)

        self.training_worker = None
        self.playback_worker = None
        self.neat_config = None

        self.history_winners = {}
        self.selected_genome = None

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        left_layout = QtWidgets.QVBoxLayout()
        self.sim_screen = ScreenWidget(1200, 800)
        left_layout.addWidget(QtWidgets.QLabel("<b>Podgląd symulacji:</b>"))
        left_layout.addWidget(self.sim_screen)
        left_layout.addStretch()
        main_layout.addLayout(left_layout, stretch=2)

        right_layout = QtWidgets.QVBoxLayout()

        param_group = QtWidgets.QGroupBox("Parametry algorytmu NEAT")
        param_form = QtWidgets.QFormLayout(param_group)

        self.spin_pop_size = QtWidgets.QSpinBox()
        self.spin_pop_size.setRange(10, 5000)
        self.spin_pop_size.setValue(150)
        param_form.addRow("Wielkość populacji:", self.spin_pop_size)

        self.spin_epochs = QtWidgets.QSpinBox()
        self.spin_epochs.setRange(1, 20000)
        self.spin_epochs.setValue(100)
        param_form.addRow("Liczba epok:", self.spin_epochs)

        self.spin_conn_add_prob = QtWidgets.QDoubleSpinBox()
        self.spin_conn_add_prob.setRange(0.0, 1.0)
        self.spin_conn_add_prob.setSingleStep(0.05)
        self.spin_conn_add_prob.setValue(0.5)
        param_form.addRow("Szansa dodania połączenia:", self.spin_conn_add_prob)

        self.spin_conn_delete_prob = QtWidgets.QDoubleSpinBox()
        self.spin_conn_delete_prob.setRange(0.0, 1.0)
        self.spin_conn_delete_prob.setSingleStep(0.05)
        self.spin_conn_delete_prob.setValue(0.5)
        param_form.addRow("Szansa usunięcia połączenia:", self.spin_conn_delete_prob)

        self.spin_compat_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_compat_thresh.setRange(0.1, 50.0)
        self.spin_compat_thresh.setSingleStep(0.2)
        self.spin_compat_thresh.setValue(3.0)
        param_form.addRow("Próg zgodności gatunkowej:", self.spin_compat_thresh)

        self.spin_survival_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_survival_thresh.setRange(0.01, 1.0)
        self.spin_survival_thresh.setSingleStep(0.05)
        self.spin_survival_thresh.setValue(0.2)
        param_form.addRow("Próg przeżywalności w gatunku:", self.spin_survival_thresh)

        self.spin_weight_mutate_power = QtWidgets.QDoubleSpinBox()
        self.spin_weight_mutate_power.setRange(0.01, 10.0)
        self.spin_weight_mutate_power.setSingleStep(0.1)
        self.spin_weight_mutate_power.setValue(0.5)
        param_form.addRow("Siła mutacji wag połączeń:", self.spin_weight_mutate_power)

        self.spin_bias_mutate_power = QtWidgets.QDoubleSpinBox()
        self.spin_bias_mutate_power.setRange(0.01, 10.0)
        self.spin_bias_mutate_power.setSingleStep(0.1)
        self.spin_bias_mutate_power.setValue(0.5)
        param_form.addRow("Siła mutacji obciążeń (biasów):", self.spin_bias_mutate_power)

        self.spin_bias_mutate_rate = QtWidgets.QDoubleSpinBox()
        self.spin_bias_mutate_rate.setRange(0.0, 1.0)
        self.spin_bias_mutate_rate.setSingleStep(0.05)
        self.spin_bias_mutate_rate.setValue(0.7)
        param_form.addRow("Szansa mutacji obciążeń (biasów):", self.spin_bias_mutate_rate)

        self.spin_enabled_mutate_rate = QtWidgets.QDoubleSpinBox()
        self.spin_enabled_mutate_rate.setRange(0.0, 1.0)
        self.spin_enabled_mutate_rate.setSingleStep(0.01)
        self.spin_enabled_mutate_rate.setValue(0.01)
        param_form.addRow("Szansa zmiany aktywności genu:", self.spin_enabled_mutate_rate)

        self.check_full_direct = QtWidgets.QCheckBox("Pełne połączenia początkowe wejść z wyjściami")
        self.check_full_direct.setChecked(True)
        param_form.addRow(self.check_full_direct)

        self.check_render = QtWidgets.QCheckBox("Renderuj symulację na żywo podczas nauki (znacznie wolniej)")
        param_form.addRow(self.check_render)

        right_layout.addWidget(param_group)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_toggle = QtWidgets.QPushButton("Start nauki")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self.handle_train_toggle)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_all)

        btn_layout.addWidget(self.btn_toggle)
        btn_layout.addWidget(self.btn_reset)
        right_layout.addLayout(btn_layout)

        self.lbl_status = QtWidgets.QLabel("Status: Gotowy")
        right_layout.addWidget(self.lbl_status)

        # Historia
        history_group = QtWidgets.QGroupBox("Historia epok (wybierz, aby zbadać)")
        history_layout = QtWidgets.QVBoxLayout(history_group)

        self.list_epochs = QtWidgets.QListWidget()
        self.list_epochs.setMinimumHeight(140)
        self.list_epochs.itemClicked.connect(self.on_epoch_selected)
        history_layout.addWidget(self.list_epochs)

        self.btn_playback = QtWidgets.QPushButton("Odtwórz wybraną epokę")
        self.btn_playback.setEnabled(False)
        self.btn_playback.clicked.connect(self.start_playback)
        history_layout.addWidget(self.btn_playback)

        right_layout.addWidget(history_group)

        # Podgląd grafu sieci
        image_group = QtWidgets.QGroupBox("Struktura sieci wybranego osobnika (kliknij, aby powiększyć)")
        image_layout = QtWidgets.QVBoxLayout(image_group)

        self.img_label = ClickableImageLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(320, 160)
        self.img_label.setText("Brak wygenerowanego obrazu")
        self.img_label.setStyleSheet(
            "border: 1px dashed #555; background-color: #1e1e1e; color: #888;"
        )

        image_layout.addWidget(self.img_label)
        right_layout.addWidget(image_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, stretch=1)

    def get_neat_params(self):
        return {
            'pop_size': self.spin_pop_size.value(),
            'initial_connection': 'full_direct' if self.check_full_direct.isChecked() else 'unconnected',
            'conn_add_prob': self.spin_conn_add_prob.value(),
            'conn_delete_prob': self.spin_conn_delete_prob.value(),
            'compatibility_threshold': self.spin_compat_thresh.value(),
            'survival_threshold': self.spin_survival_thresh.value(),
            'weight_mutate_power': self.spin_weight_mutate_power.value(),
            'bias_mutate_power': self.spin_bias_mutate_power.value(),
            'bias_mutate_rate': self.spin_bias_mutate_rate.value(),
            'enabled_mutate_rate': self.spin_enabled_mutate_rate.value(),
        }

    def handle_train_toggle(self, checked):
        if checked:
            self.start_training()
        else:
            self.stop_training()

    def start_training(self):
        self.stop_playback()

        self.history_winners.clear()
        self.list_epochs.clear()
        self.selected_genome = None
        self.btn_playback.setEnabled(False)

        neat_params = self.get_neat_params()

        self.training_worker = TrainingWorker(
            config_path='neat-config.ini',
            neat_params=neat_params,
            total_epochs=self.spin_epochs.value(),
            render_enabled=self.check_render.isChecked(),
        )
        self.training_worker.frame_ready.connect(self.sim_screen.update_frame)
        self.training_worker.status_changed.connect(self.lbl_status.setText)
        self.training_worker.epoch_recorded.connect(self.record_epoch)
        self.training_worker.training_done.connect(self.on_training_done)

        self.btn_toggle.setChecked(True)
        self.btn_toggle.setText("Stop")
        self.list_epochs.setEnabled(False)
        self.set_params_enabled(False)

        self.training_worker.start()

    def stop_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.lbl_status.setText("Zatrzymywanie...")
            self.training_worker.stop()

    def record_epoch(self, epoch_num, best_fitness, genome_snapshot):
        self.history_winners[epoch_num] = genome_snapshot
        item_text = f"Epoka {epoch_num:3d} | Wynik (Fitness): {best_fitness:8.2f}"
        item = QtWidgets.QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, epoch_num)
        self.list_epochs.addItem(item)
        self.list_epochs.scrollToBottom()

    def on_training_done(self, winner, config):
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Start nauki")
        self.set_params_enabled(True)
        self.list_epochs.setEnabled(True)
        self.neat_config = config

        if self.list_epochs.count() > 0:
            last_item = self.list_epochs.item(self.list_epochs.count() - 1)
            self.list_epochs.setCurrentItem(last_item)
            self.on_epoch_selected(last_item)

    def on_epoch_selected(self, item):
        self.stop_playback()

        epoch_num = item.data(Qt.ItemDataRole.UserRole)
        genome = self.history_winners.get(epoch_num)
        if not genome or not self.neat_config:
            return

        self.selected_genome = genome
        self.btn_playback.setEnabled(True)
        self.lbl_status.setText(f"Wybrano osobnika z epoki {epoch_num}")

        try:
            filename = f"network_epoch_{epoch_num}"
            visualize.draw_net(
                self.neat_config, genome, view=False, filename=filename
            )
            svg_file = f"{filename}.svg"
            net_path = svg_file if os.path.exists(svg_file) else filename
            self.display_image(net_path)
        except Exception as e:
            print(f"Błąd generowania wykresu sieci: {e}")

    def start_playback(self):
        if self.playback_worker and self.playback_worker.isRunning():
            self.stop_playback()
            return

        if not self.selected_genome or not self.neat_config:
            return

        self.stop_playback()

        self.playback_worker = PlaybackWorker(
            self.selected_genome, self.neat_config
        )
        self.playback_worker.frame_ready.connect(self.sim_screen.update_frame)
        self.playback_worker.playback_done.connect(self.on_playback_done)

        self.btn_toggle.setEnabled(False)
        self.list_epochs.setEnabled(False)
        self.btn_playback.setText("Stop odtwarzania")
        self.lbl_status.setText("Odtwarzanie wybranego osobnika...")
        self.playback_worker.start()

    def stop_playback(self):
        if self.playback_worker and self.playback_worker.isRunning():
            self.lbl_status.setText("Zatrzymywanie odtwarzania...")
            self.playback_worker.stop()
            try:
                self.playback_worker.frame_ready.disconnect()
                self.playback_worker.playback_done.disconnect()
            except RuntimeError:
                pass
            self.playback_worker.wait()
            self.on_playback_done()

    def on_playback_done(self):
        self.btn_toggle.setEnabled(True)
        self.btn_playback.setEnabled(True)
        self.btn_playback.setText("Odtwórz wybraną epokę")
        self.list_epochs.setEnabled(True)
        self.lbl_status.setText("Gotowy.")

    def reset_all(self):
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait()

        self.stop_playback()

        self.history_winners.clear()
        self.selected_genome = None
        self.neat_config = None

        self.sim_screen.clear_screen()
        self.list_epochs.clear()
        self.img_label.clear()
        self.img_label.current_pixmap = None
        self.img_label.current_file_path = None
        self.img_label.setText("Brak wygenerowanego obrazu")
        self.lbl_status.setText("Status: Zresetowano")

        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Start nauki")
        self.btn_playback.setEnabled(False)
        self.btn_playback.setText("Odtwórz wybraną epokę")
        self.list_epochs.setEnabled(True)
        self.set_params_enabled(True)

    def set_params_enabled(self, enabled: bool):
        self.spin_pop_size.setEnabled(enabled)
        self.spin_epochs.setEnabled(enabled)
        self.spin_conn_add_prob.setEnabled(enabled)
        self.spin_conn_delete_prob.setEnabled(enabled)
        self.spin_compat_thresh.setEnabled(enabled)
        self.spin_survival_thresh.setEnabled(enabled)
        self.spin_weight_mutate_power.setEnabled(enabled)
        self.spin_bias_mutate_power.setEnabled(enabled)
        self.spin_bias_mutate_rate.setEnabled(enabled)
        self.spin_enabled_mutate_rate.setEnabled(enabled)
        self.check_full_direct.setEnabled(enabled)
        self.check_render.setEnabled(enabled)

    def display_image(self, file_path):
        self.img_label.set_custom_image(file_path)


if __name__ == "__main__":
    mp.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())