# -*- coding: utf-8 -*-
"""
automation.py
Fluxo por ciclo:
1) Procurar 'bm.png'
   - Se achou: procurar 'bm-buy.png'
       - Se achar: clicar no centro-inferior, ALERTA simples e PAUSAR.
       - Se NÃO achar: ALERTA forte e PAUSAR imediatamente.
   - Se não achou 'bm.png', segue para passo 2.
2) Procurar 'mystic.png'
   - Mesmo fluxo com 'mystic-buy.png'.

Extras:
- Loga "Janela em {W}x{H} @ {L},{T}" quando mapeia/muda.
- Se a janela não estiver em 922x549, tenta redimensionar para 922x549.
- Captura do conteúdo da janela e matching multiescala.
- Pastas de imagens: ./src (bm.png, bm-buy.png, mystic.png, mystic-buy.png)

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

TARGET_W, TARGET_H = 922, 549        # tamanho desejado da janela

# Templates (em ./src)
T_BM = "bm.png"
T_BM_BUY = "bm-buy.png"
T_MY = "mystic.png"
T_MY_BUY = "mystic-buy.png"

# Matching (ajustável)
SCALES = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
TM_THRESHOLD = 0.82
DEBUG_SAVE = False  # opcional: salva ./src/debug/last_window.png


# ===== Caminhos =====
def get_base_path() -> str:
    """Retorna o diretório base (script ou frozen)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def src_path(filename: str) -> str:
    """./src/<filename>"""
    return os.path.join(get_base_path(), "src", filename)


def ensure_debug_dir() -> str:
    """Cria/retorna ./src/debug"""
    d = os.path.join(get_base_path(), "src", "debug")
    os.makedirs(d, exist_ok=True)
    return d


# ===== Janela alvo =====
def find_epic_seven_window():
    """Retorna a primeira janela cujo título contém 'epic seven' (ou None)."""
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
    """Traz a janela para o foco (ignora falhas)."""
    try:
        if win.isMinimized:
            win.restore()
        win.activate()
        time.sleep(0.05)
    except Exception:
        pass


def get_window_region(win) -> Optional[Tuple[int, int, int, int]]:
    """Retorna (left, top, width, height) inteiros da janela."""
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
    Garante que a janela esteja em TARGET_W x TARGET_H.
    Tenta redimensionar e retorna a região atualizada; None se falhar.
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
                # Alguns ambientes não suportam resize; segue com o tamanho atual
                pass
            region = get_window_region(win)
            if region:
                L, T, W, H = region
                log_fn(f"Janela em {W}x{H} @ {L},{T}")
            return region
        else:
            # Já está no tamanho desejado; opcionalmente logar ao mudar (feito no loop)
            return region
    except Exception:
        return get_window_region(win)


# ===== Captura =====
def _is_valid_frame(frame: Optional["np.ndarray"]) -> bool:
    """Checa se o frame é válido (não vazio/preto)."""
    if frame is None:
        return False
    if frame.size == 0:
        return False
    if np is not None and np.var(frame) < 5.0:
        return False
    return True


def capture_with_pyautogui(region: Tuple[int, int, int, int]) -> Optional["np.ndarray"]:
    """Captura com pyautogui (RGB -> BGR)."""
    if pag is None or np is None:
        return None
    try:
        img = pag.screenshot(region=region)  # PIL
        arr = np.array(img)[:, :, ::-1].copy()  # BGR
        return arr
    except Exception:
        return None


def capture_with_mss(region: Tuple[int, int, int, int]) -> Optional["np.ndarray"]:
    """Captura com mss (BGRA -> BGR)."""
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
    """Tenta pyautogui e, se falhar, usa mss."""
    bgr = capture_with_pyautogui(region)
    if not _is_valid_frame(bgr):
        bgr = capture_with_mss(region)
    if not _is_valid_frame(bgr):
        return None
    return bgr


# ===== Templates & Matching =====
def load_templates() -> Dict[str, "np.ndarray"]:
    """Carrega templates em tons de cinza: retorna {nome: img_gray}."""
    items: Dict[str, "np.ndarray"] = {}
    if cv2 is None:
        return items
    for name in (T_BM, T_BM_BUY, T_MY, T_MY_BUY):
        path = src_path(name)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                items[name] = img
    return items


def tm_best_match(win_gray: "np.ndarray",
                  tmpl_gray: "np.ndarray",
                  scales=SCALES) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Template Matching multiescala.
    Retorna (x, y, w, h, score) do melhor match ou None.
    """
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


# ===== Clique & Alerta =====
def click_bottom_center(region: Tuple[int, int, int, int],
                        rect: Tuple[int, int, int, int],
                        margin_px: int = 3) -> bool:
    """
    Clica no centro-inferior do retângulo detectado.
    - region: (L, T, W, H) da janela
    - rect: (x, y, w, h) relativo à janela
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


def beep_alert(level: str = "info"):
    """
    Emite um alerta sonoro:
    - 'info': beep curto
    - 'alert': padrão mais chamativo (vários beeps)
    """
    try:
        if winsound is not None:
            if level == "alert":
                for f, d in ((1000, 160), (700, 160), (1200, 240), (900, 200)):
                    winsound.Beep(f, d)
            else:
                winsound.Beep(880, 180)
        else:
            # Fallback mínimo
            if level == "alert":
                print("\a\a\a\a")
            else:
                print("\a")
    except Exception:
        pass


def pause_now(pause_event, log_fn, level: str):
    """Aciona pausa, toca alerta e loga 'PAUSADO'."""
    pause_event.set()
    beep_alert(level=level)
    log_fn("PAUSADO")


# ===== Passo unitário do fluxo =====
def step_flow(region, frame_bgr, templates, prim_name, buy_name, log_fn, pause_event) -> bool:
    """
    Executa:
      procurar prim_name -> se achou, procurar buy_name ->
         - se achou: clicar, ALERTA info e PAUSAR (retorna True)
         - se não achou: ALERTA forte e PAUSAR (retorna True)
    Retorna True quando entrou em pausa; False caso contrário.
    """
    if cv2 is None or np is None:
        return False

    win_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Primário
    prim_tmpl = templates.get(prim_name)
    if prim_tmpl is not None:
        best = tm_best_match(win_gray, prim_tmpl)
        if best and best[4] >= TM_THRESHOLD:
            log_fn(f"{prim_name.split('.')[0]}: encontrado")

            # Botão buy
            buy_tmpl = templates.get(buy_name)
            if buy_tmpl is not None:
                best_buy = tm_best_match(win_gray, buy_tmpl)
                if best_buy and best_buy[4] >= TM_THRESHOLD:
                    x, y, w, h, _ = best_buy
                    if click_bottom_center(region, (x, y, w, h)):
                        log_fn(f"{buy_name.split('.')[0]}: encontrado (clicou)")
                        pause_now(pause_event, log_fn, level="info")
                        return True
                    else:
                        # Falha no clique: ainda assim, pause com alerta forte para intervenção
                        log_fn(f"{buy_name.split('.')[0]}: encontrado (falha no clique)")
                        pause_now(pause_event, log_fn, level="alert")
                        return True
                # buy não encontrado -> pausa imediata (alerta forte)
                log_fn(f"{buy_name.split('.')[0]}: não encontrado")
                pause_now(pause_event, log_fn, level="alert")
                return True
            else:
                log_fn(f"{buy_name.split('.')[0]}: não encontrado")
                pause_now(pause_event, log_fn, level="alert")
                return True
        else:
            log_fn(f"{prim_name.split('.')[0]}: não encontrado")
            return False
    else:
        log_fn(f"{prim_name.split('.')[0]}: não encontrado")
        return False


# ===== Loop principal (chamado pela GUI) =====
def run(stop_event,
        pause_event,
        intervalo_s: float,
        log_fn):
    """
    Por ciclo:
      - Garante/foca janela 'Epic Seven'; ajusta para 922x549 se necessário.
      - Tenta fluxo BM; se pausar, aguarda retomar.
      - Senão, tenta fluxo MYSTIC; se pausar, aguarda retomar.
    """
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

            # Região e ajuste de tamanho
            region = get_window_region(win)
            if region:
                # Ajusta para 922x549 se necessário (e loga tamanho atual)
                if (region[2], region[3]) != (TARGET_W, TARGET_H):
                    region = ensure_window_size(win, log_fn)
                # Se mudou ou foi forçada, loga dimensão
                if region and region != last_region:
                    L, T, W, H = region
                    log_fn(f"Janela em {W}x{H} @ {L},{T}")
                    last_region = region

                if region:
                    frame = capture_window_bgr(region)
                    if frame is not None:
                        if DEBUG_SAVE and cv2 is not None:
                            try:
                                d = ensure_debug_dir()
                                cv2.imwrite(os.path.join(d, "last_window.png"), frame)
                            except Exception:
                                pass

                        # Passo 1: BM
                        if step_flow(region, frame, templates, T_BM, T_BM_BUY, log_fn, pause_event):
                            # entrou em pausa; espera próximo ciclo
                            _sleep_controlado(intervalo_s, stop_event, pause_event)
                            continue

                        # Passo 2: MYSTIC
                        if step_flow(region, frame, templates, T_MY, T_MY_BUY, log_fn, pause_event):
                            _sleep_controlado(intervalo_s, stop_event, pause_event)
                            continue

        # Espera entre ciclos
        _sleep_controlado(intervalo_s, stop_event, pause_event)


def _sleep_controlado(total: float, stop_event, pause_event, step: float = 0.1):
    """Dorme em passos curtos respeitando pausa e parada."""
    elapsed = 0.0
    while elapsed < total and not stop_event.is_set():
        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(step)
        if stop_event.is_set():
            break
        time.sleep(step)
        elapsed += step
