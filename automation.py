# -*- coding: utf-8 -*-
"""
automation.py

Fluxo por ciclo:
1) Procurar 'bm.png'
   1.1) Se achou: procurar 'bm-buy.png' e clicar no MEIO-PRA-BAIXO do botão.
   1.2) Procurar 'cancel-buy.png' e clicar moderadamente À DIREITA de onde encontrou.
   1.3) Se NÃO achou 'bm-buy.png' (após achar 'bm.png'): PAUSAR imediatamente com alerta.

2) Procurar 'mystic.png'
   (mesmos passos que 1)

3) Scrolar para baixo N vezes, repetindo os passos 1 e 2 a cada scroll.

4) PAUSAR ao final do ciclo e aguardar retomar.

Logs simples:
- "Janela em {W}x{H} @ {L},{T}" quando mapeia/muda
- "bm: encontrado" / "bm: não encontrado"
- "bm-buy: encontrado (clicou)" / "bm-buy: não encontrado -> pausar"
- "cancel-buy: clicou à direita" / "cancel-buy: não encontrado"
- "mystic: encontrado" / "mystic: não encontrado"
- "mystic-buy: encontrado (clicou)" / "mystic-buy: não encontrado -> pausar"
- "SCROLL {i}/{N}"
- "PAUSADO"

Requisitos:
    pip install pyautogui pillow mss opencv-python numpy pygetwindow
"""

import os
import sys
import time
from typing import Optional, Tuple, Dict

# ===== Dependências =====
try:
    import pyautogui as pag
except Exception:
    pag = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    from mss import mss
except Exception:
    mss = None

try:
    import pygetwindow as gw
except Exception:
    gw = None

# Beep (Windows)
try:
    import winsound
except Exception:
    winsound = None


# ===== Constantes =====
WINDOW_TITLE_SUBSTR = "epic seven"   # título fixo (case-insensitive)
FOCAR_JANELA_CADA_LOOP = True

TARGET_W, TARGET_H = 922, 549        # tamanho desejado da janela (ajuste automático)

# Templates (em ./src)
T_BM        = "bm.png"
T_BM_BUY    = "bm-buy.png"
T_MY        = "mystic.png"
T_MY_BUY    = "mystic-buy.png"
T_CANCEL    = "cancel-buy.png"

# Matching
SCALES       = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
TM_THRESHOLD = 0.82

# Scrolling
SCROLL_STEPS       = 7     # quantas vezes scrolar por ciclo
SCROLL_AMOUNT_LINE = 60   # quanto scrolar por passo (valor positivo; usaremos negativo para descer)

# Cliques
BOTTOM_MARGIN_PX   = 3     # margem para "meio-pra-baixo"
RIGHT_OFFSET_RATIO = 0.6   # quanto ir para a direita a partir do centro do 'cancel-buy' (em largura do template)

DEBUG_SAVE = False  # se True, salva ./src/debug/last_window.png do recorte


# ===== Caminhos =====
def get_base_path() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def src_path(filename: str) -> str:
    return os.path.join(get_base_path(), "src", filename)


def ensure_debug_dir() -> str:
    d = os.path.join(get_base_path(), "src", "debug")
    os.makedirs(d, exist_ok=True)
    return d


# ===== Janela alvo =====
def find_epic_seven_window():
    if gw is None:
        return None
    target = WINDOW_TITLE_SUBSTR
    for t in gw.getAllTitles():
        if target in (t or "").lower():
            wins = gw.getWindowsWithTitle(t)
            if wins:
                return wins[0]
    return None


def bring_to_front(win) -> None:
    try:
        if win.isMinimized:
            win.restore()
        win.activate()
        time.sleep(0.05)
    except Exception:
        pass


def get_window_region(win) -> Optional[Tuple[int, int, int, int]]:
    try:
        L, T = int(win.left), int(win.top)
        W, H = int(win.width), int(win.height)
        if W > 0 and H > 0:
            return (L, T, W, H)
    except Exception:
        pass
    return None


def ensure_window_size(win, log_fn) -> Optional[Tuple[int, int, int, int]]:
    """
    Garante 922x549; tenta resizeTo; retorna região atualizada.
    """
    try:
        region = get_window_region(win)
        if not region:
            return None
        L, T, W, H = region
        if (W, H) != (TARGET_W, TARGET_H):
            try:
                win.resizeTo(TARGET_W, TARGET_H)
                time.sleep(0.12)
            except Exception:
                pass
            region = get_window_region(win)
            if region:
                L, T, W, H = region
                log_fn(f"Janela em {W}x{H} @ {L},{T}")
            return region
        return region
    except Exception:
        return get_window_region(win)


# ===== Captura =====
def _is_valid_frame(frame: Optional["np.ndarray"]) -> bool:
    if frame is None:
        return False
    if frame.size == 0:
        return False
    if np is not None and np.var(frame) < 5.0:
        return False
    return True


def capture_with_pyautogui(region: Tuple[int, int, int, int]) -> Optional["np.ndarray"]:
    if pag is None or np is None:
        return None
    try:
        img = pag.screenshot(region=region)  # PIL
        arr = np.array(img)[:, :, ::-1].copy()  # BGR
        return arr
    except Exception:
        return None


def capture_with_mss(region: Tuple[int, int, int, int]) -> Optional["np.ndarray"]:
    if mss is None or np is None:
        return None
    L, T, W, H = region
    try:
        with mss() as sct:
            shot = sct.grab({"left": L, "top": T, "width": W, "height": H})
            arr = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((shot.height, shot.width, 3))
            return arr.copy()
    except Exception:
        return None


def capture_window_bgr(region: Tuple[int, int, int, int]) -> Optional["np.ndarray"]:
    bgr = capture_with_pyautogui(region)
    if not _is_valid_frame(bgr):
        bgr = capture_with_mss(region)
    if not _is_valid_frame(bgr):
        return None
    return bgr


# ===== Templates & Matching =====
def load_templates() -> Dict[str, "np.ndarray"]:
    items: Dict[str, "np.ndarray"] = {}
    if cv2 is None:
        return items
    for name in (T_BM, T_BM_BUY, T_MY, T_MY_BUY, T_CANCEL):
        path = src_path(name)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                items[name] = img
    return items


def tm_best_match(win_gray: "np.ndarray",
                  tmpl_gray: "np.ndarray",
                  scales=SCALES) -> Optional[Tuple[int, int, int, int, float]]:
    best = None
    for s in scales:
        w = max(1, int(tmpl_gray.shape[1] * s))
        h = max(1, int(tmpl_gray.shape[0] * s))
        t = cv2.resize(tmpl_gray, (w, h), interpolation=cv2.INTER_AREA)
        if win_gray.shape[0] < h or win_gray.shape[1] < w:
            continue
        res = cv2.matchTemplate(win_gray, t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or max_val > best[4]:
            best = (max_loc[0], max_loc[1], w, h, max_val)
    return best


# ===== Cliques utilitários =====
def click_bottom_center(region: Tuple[int, int, int, int],
                        rect: Tuple[int, int, int, int],
                        margin_px: int = BOTTOM_MARGIN_PX) -> bool:
    """
    Clica no centro-inferior do retângulo detectado relativo à janela.
    """
    if pag is None:
        return False
    L, T, _W, _H = region
    x, y, w, h = rect
    abs_x = L + x + (w // 2)
    abs_y = T + y + h - 1 - max(0, margin_px)
    try:
        pag.moveTo(abs_x, abs_y)
        pag.click()
        return True
    except Exception:
        return False


def click_right_of(region: Tuple[int, int, int, int],
                   rect: Tuple[int, int, int, int],
                   offset_ratio: float = RIGHT_OFFSET_RATIO) -> bool:
    """
    Clica moderadamente à direita do centro do retângulo (ex.: próximo ao botão 'confirmar').
    """
    if pag is None:
        return False
    L, T, _W, _H = region
    x, y, w, h = rect
    cx = x + (w // 2)
    cy = y + (h // 2)
    dx = int(w * max(0.0, offset_ratio))  # deslocamento à direita
    abs_x = L + cx + dx
    abs_y = T + cy
    try:
        pag.moveTo(abs_x, abs_y)
        pag.click()
        return True
    except Exception:
        return False


def center_of(region: Tuple[int, int, int, int]) -> Tuple[int, int]:
    L, T, W, H = region
    return (L + W // 2, T + H // 2)


def beep(level: str = "info"):
    try:
        if winsound is not None:
            if level == "alert":
                for f, d in ((1000, 160), (700, 160), (1200, 240), (900, 200)):
                    winsound.Beep(f, d)
            else:
                winsound.Beep(880, 180)
        else:
            print("\a")
    except Exception:
        pass


def pause_now(pause_event, log_fn, level: str):
    pause_event.set()
    beep(level=level)
    log_fn("PAUSADO")


# ===== Passos 1 e 2 (genérico) =====
def run_flow_for_anchor(region, frame_bgr, templates, anchor_name, buy_name, log_fn, pause_event) -> bool:
    """
    Executa a sequência para uma âncora (bm/mystic):
      - encontrar âncora
      - se encontrou: procurar buy, clicar no meio-pra-baixo
      - procurar cancel-buy e clicar à direita (se existir)
      - se NÃO encontrou buy (após achar âncora): PAUSAR e retornar True
    Retorna:
      True  -> entrou em pausa por falha de buy
      False -> não pausou (segue fluxo)
    """
    if cv2 is None or np is None:
        return False

    win_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # âncora
    anchor_t = templates.get(anchor_name)
    if anchor_t is None:
        log_fn(f"{anchor_name.split('.')[0]}: não encontrado")
        return False

    best_anchor = tm_best_match(win_gray, anchor_t)
    if not (best_anchor and best_anchor[4] >= TM_THRESHOLD):
        log_fn(f"{anchor_name.split('.')[0]}: não encontrado")
        return False

    log_fn(f"{anchor_name.split('.')[0]}: encontrado")

    # buy
    buy_t = templates.get(buy_name)
    if buy_t is None:
        log_fn(f"{buy_name.split('.')[0]}: não encontrado -> pausar")
        pause_now(pause_event, log_fn, level="alert")
        return True

    best_buy = tm_best_match(win_gray, buy_t)
    if not (best_buy and best_buy[4] >= TM_THRESHOLD):
        log_fn(f"{buy_name.split('.')[0]}: não encontrado -> pausar")
        pause_now(pause_event, log_fn, level="alert")
        return True

    x, y, w, h, _ = best_buy
    if click_bottom_center(region, (x, y, w, h)):
        log_fn(f"{buy_name.split('.')[0]}: encontrado (clicou)")
        time.sleep(0.12)  # pequena espera para abrir caixa de confirmação
    else:
        log_fn(f"{buy_name.split('.')[0]}: encontrado (falha no clique) -> pausar")
        pause_now(pause_event, log_fn, level="alert")
        return True
    time.sleep(3)
    # cancel-buy (opcional)
    cancel_t = templates.get(T_CANCEL)
    if cancel_t is not None:
        best_cancel = tm_best_match(win_gray, cancel_t)
        if best_cancel and best_cancel[4] >= TM_THRESHOLD:
            cx, cy, cw, ch, _ = best_cancel
            if click_right_of(region, (cx, cy, cw, ch)):
                log_fn("cancel-buy: clicou à direita")
            else:
                log_fn("cancel-buy: falha ao clicar à direita")
        else:
            log_fn("cancel-buy: não encontrado 1")
            log_fn(T_CANCEL)
            pause_now(pause_event, log_fn, level="alert")
    else:
        log_fn("cancel-buy: imagem não encontrada")
        pause_now(pause_event, log_fn, level="alert")

    return False


# ===== Loop principal =====
def run(stop_event,
        pause_event,
        intervalo_s: float,
        log_fn):
    templates = load_templates()
    last_region = None

    while not stop_event.is_set():
        # respeita pausa externa
        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        if stop_event.is_set():
            break

        win = find_epic_seven_window()
        if win is not None:
            if FOCAR_JANELA_CADA_LOOP:
                bring_to_front(win)
                time.sleep(0.05)

            region = get_window_region(win)
            if region:
                # ajusta janela para 922x549 se necessário e loga dimensões
                if (region[2], region[3]) != (TARGET_W, TARGET_H):
                    region = ensure_window_size(win, log_fn)
                if region and region != last_region:
                    L, T, W, H = region
                    log_fn(f"Janela em {W}x{H} @ {L},{T}")
                    last_region = region

                if region:
                    # ciclo base (sem scroll)
                    frame = capture_window_bgr(region)
                    if frame is not None:
                        # passo 1: bm
                        if run_flow_for_anchor(region, frame, templates, T_BM, T_BM_BUY, log_fn, pause_event):
                            _sleep_controlado(intervalo_s, stop_event, pause_event)
                            continue
                        # passo 2: mystic
                        if run_flow_for_anchor(region, frame, templates, T_MY, T_MY_BUY, log_fn, pause_event):
                            _sleep_controlado(intervalo_s, stop_event, pause_event)
                            continue

                        # 3) scrolar para baixo e repetir
                        for i in range(1, SCROLL_STEPS + 1):
                            if stop_event.is_set() or pause_event.is_set():
                                break
                            log_fn(f"SCROLL {i}/{SCROLL_STEPS}")
                            try:
                                cx, cy = center_of(region)
                                pag.moveTo(cx, cy)
                                pag.scroll(-abs(SCROLL_AMOUNT_LINE))
                            except Exception:
                                pass
                            time.sleep(0.15)
                            frame2 = capture_window_bgr(region)
                            if frame2 is None:
                                continue
                            if run_flow_for_anchor(region, frame2, templates, T_BM, T_BM_BUY, log_fn, pause_event):
                                break
                            if run_flow_for_anchor(region, frame2, templates, T_MY, T_MY_BUY, log_fn, pause_event):
                                break

                        # 4) pausa ao final do ciclo
                        if not pause_event.is_set():
                            pause_now(pause_event, log_fn, level="info")

        # espera entre ciclos
        _sleep_controlado(intervalo_s, stop_event, pause_event)


def _sleep_controlado(total: float, stop_event, pause_event, step: float = 0.1):
    elapsed = 0.0
    while elapsed < total and not stop_event.is_set():
        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(step)
        if stop_event.is_set():
            break
        time.sleep(step)
        elapsed += step
