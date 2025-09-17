# -*- coding: utf-8 -*-
"""
controle_gui.py
GUI com:
- Botão principal: Iniciar -> Pausar -> Retomar
- Botão Parar
- Área de logs (somente mensagens vindas da automação)
- Campo único de configuração: Intervalo (s)

Títulos e imagens são fixos dentro de automation.py:
  - Título da janela: "Epic Seven"
  - Imagens: ./scr/mystic.png e ./scr/bm.png
"""

import threading
import queue
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

import automation


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Controle de Automação")
        self.geometry("820x520")
        self.minsize(700, 420)

        # Estado: 'stopped' | 'running' | 'paused'
        self._state = 'stopped'
        self._worker_thread = None
        self._stop_event = None
        self._pause_event = None

        # Fila de logs (automação -> GUI)
        self._log_queue = queue.Queue()

        # Configuração: apenas intervalo
        self.intervalo_var = tk.DoubleVar(value=1.0)

        self._build_ui()
        self.after(80, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self):
        topbar = ttk.Frame(self, padding=(10, 10))
        topbar.pack(side=tk.TOP, fill=tk.X)

        self.toggle_btn = ttk.Button(topbar, text="Iniciar", command=self._toggle_main, width=16)
        self.toggle_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(topbar, text="Parar", command=self._stop_worker, width=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(topbar, text="Limpar Logs", command=self._clear_logs).pack(side=tk.LEFT, padx=(10, 0))

        self.status_lbl = ttk.Label(topbar, text="Status: parado")
        self.status_lbl.pack(side=tk.RIGHT)

        config = ttk.LabelFrame(self, text="Configurações", padding=(10, 10))
        config.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(config, text="Intervalo (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(config, textvariable=self.intervalo_var, width=10).grid(row=0, column=1, sticky="w", padx=(5, 0))

        log_frame = ttk.LabelFrame(self, text="Logs de Execução", padding=(6, 6))
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, height=18, state=tk.NORMAL)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.tag_config("INFO", foreground="#111111")

    # ---------- Logs ----------
    def _log(self, message: str):
        now = datetime.now().strftime("%H:%M:%S")
        self._log_queue.put(f"{now} | {message}")

    def _poll_log_queue(self):
        """
        Consome a fila de logs e escreve no Text. Também detecta 'PAUSADO'
        para refletir o auto-pause na interface (botão vira 'Retomar').
        """
        import queue as _q
        try:
            while True:
                line = self._log_queue.get_nowait()
                # escreve no painel
                self.log_text.insert(tk.END, line + "\n", "INFO")
                self.log_text.see(tk.END)

                # REFLEXO DO AUTO-PAUSE:
                # A automação escreve exatamente 'PAUSADO' na mensagem.
                # Como a GUI prefixa horário (ex.: "12:34:56 | PAUSADO"),
                # checamos por substring.
                if "PAUSADO" in line and self._state == 'running':
                    self._state = 'paused'
                    self._update_controls_paused()
        except _q.Empty:
            pass

        # agenda a próxima leitura
        self.after(80, self._poll_log_queue)


    # ---------- Botões ----------
    def _toggle_main(self):
        if self._state == 'stopped':
            self._start_worker()
        elif self._state == 'running':
            self._pause_worker()
        elif self._state == 'paused':
            self._resume_worker()

    def _start_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        try:
            intervalo = float(self.intervalo_var.get())
            if intervalo <= 0:
                raise ValueError
        except Exception:
            return

        import threading
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        args = (self._stop_event, self._pause_event, intervalo, self._log)
        self._worker_thread = threading.Thread(target=automation.run, args=args, daemon=True)
        self._worker_thread.start()

        self._state = 'running'
        self._update_controls_running()

    def _pause_worker(self):
        if self._state != 'running' or not self._pause_event:
            return
        self._pause_event.set()
        self._state = 'paused'
        self._update_controls_paused()

    def _resume_worker(self):
        if self._state != 'paused' or not self._pause_event:
            return
        self._pause_event.clear()
        self._state = 'running'
        self._update_controls_running()

    def _stop_worker(self):
        if self._state == 'stopped' or not self._stop_event:
            return
        self.stop_btn.config(state=tk.DISABLED)
        self.toggle_btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="Status: encerrando...")

        self._stop_event.set()
        if self._pause_event and self._pause_event.is_set():
            self._pause_event.clear()
        self.after(120, self._watch_thread_stop)

    def _watch_thread_stop(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self.after(120, self._watch_thread_stop)
        else:
            self._state = 'stopped'
            self.toggle_btn.config(text="Iniciar", state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_lbl.config(text="Status: parado")
            self._stop_event = None
            self._pause_event = None
            self._worker_thread = None

    # ---------- Helpers ----------
    def _update_controls_running(self):
        self.toggle_btn.config(text="Pausar", state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text="Status: em execução")

    def _update_controls_paused(self):
        self.toggle_btn.config(text="Retomar", state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text="Status: pausado")

    def _clear_logs(self):
        self.log_text.delete("1.0", tk.END)

    def _on_close(self):
        if self._state != 'stopped':
            self._stop_worker()
            self.after(250, self._try_destroy_after_stop)
        else:
            self.destroy()

    def _try_destroy_after_stop(self):
        if self._state == 'stopped':
            self.destroy()
        else:
            self.after(200, self._try_destroy_after_stop)


if __name__ == "__main__":
    app = App()
    app.mainloop()
