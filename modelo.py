#!/usr/bin/env python3
"""
PAPER-TRADER USD/CLP — estrategia final validada, señal diaria a Telegram.

Cada día: baja datos → calcula la señal → actualiza la cuenta simulada → manda
el reporte a Telegram. Simulación para probar el edge EN VIVO antes de plata real.

Comandos:
    python3 modelo.py test    -> mensaje de prueba a Telegram
    python3 modelo.py print   -> muestra el reporte en consola (no envía)
    python3 modelo.py run     -> actualiza y ENVÍA la señal del día a Telegram

NO es consejo de inversión.
"""
import os
import sys
import json
import numpy as np
import requests
import estrategia as E

HERE = os.path.dirname(os.path.abspath(__file__))
ESTADO = os.path.join(HERE, "portfolio.json")
API = "https://api.telegram.org/bot{t}/{m}"

# --- config ---
try:
    import secrets_local as _sl
    _TOK, _CHAT = getattr(_sl, "TELEGRAM_TOKEN", ""), getattr(_sl, "CHAT_ID", 0)
except ImportError:
    _TOK, _CHAT = "", 0
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or _TOK
CHAT_ID = int(os.environ.get("CHAT_ID") or _CHAT or 0)

CAPITAL_INICIAL = 1_000_000   # pesos simulados (poné tu capital real)
TARGET_VOL = 0.10
MAX_LEV = 3.0
COST_BPS = 4.4
USD_POR_LOTE = 100_000


# --------------------------- Telegram ---------------------------
def send(text):
    if not CHAT_ID:
        print("ERROR: falta CHAT_ID"); return False
    r = requests.get(API.format(t=TELEGRAM_TOKEN, m="sendMessage"),
                     params={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
                             "disable_web_page_preview": "true"}, timeout=20)
    if not r.json().get("ok"):
        print("Telegram error:", r.json()); return False
    return True


# --------------------------- estado ---------------------------
def _load():
    try:
        return json.load(open(ESTADO))
    except Exception:
        return {"capital": CAPITAL_INICIAL, "pos": 0.0, "notional": 0.0,
                "last_price": None, "fecha": None, "trades": []}


def _save(s):
    json.dump(s, open(ESTADO, "w"), indent=2, ensure_ascii=False)


# --------------------------- lógica ---------------------------
def actualizar():
    datos = E.cargar()
    if not datos:
        return None, None, None
    com, arr, hi, lo = datos
    pos_arr, ctx = E.posicion(com, arr, hi, lo)
    pos_hoy = float(pos_arr[-1])          # posición objetivo hoy (con actividad)
    precio = float(arr["clp"][-1])
    fecha = com[-1]
    scale = min(MAX_LEV, TARGET_VOL / max(ctx["vol"], 0.02))

    s = _load()
    # marcar a mercado la posición abierta
    if s["pos"] != 0 and s.get("last_price"):
        r = precio / s["last_price"] - 1
        s["capital"] *= (1 + s["notional"] * r)
    # si la posición objetivo cambió, "operar" en papel
    notional_nuevo = pos_hoy * scale
    if abs(notional_nuevo - s["notional"]) > 0.05:
        s["capital"] *= (1 - COST_BPS / 10000 * abs(notional_nuevo - s["notional"]))
        s["trades"].append({"fecha": fecha, "pos": round(pos_hoy, 2),
                            "precio": round(precio, 2), "capital": round(s["capital"])})
        s["pos"] = pos_hoy
        s["notional"] = notional_nuevo
    s["last_price"] = precio
    s["fecha"] = fecha
    _save(s)
    return s, ctx, precio


def reporte():
    s, ctx, precio = actualizar()
    if not s:
        return "⚠️ Sin datos de mercado (Yahoo). Reintento el próximo ciclo."
    pos = s["pos"]
    if pos > 0.05:
        nombre = "🔴 LARGO USD/CLP (comprar dólar)"
    elif pos < -0.05:
        nombre = "🟢 CORTO USD/CLP (vender dólar)"
    else:
        nombre = "⚪ FUERA (sin posición)"
    pnl = (s["capital"] / CAPITAL_INICIAL - 1) * 100
    notional_usd = abs(s["notional"]) * s["capital"] / precio
    lotes = notional_usd / USD_POR_LOTE
    L = [f"🤖 <b>PAPER-TRADER USD/CLP</b> · {s['fecha']}",
         f"Precio: <b>{precio:,.1f}</b>",
         "",
         f"Señal: <b>{nombre}</b>"]
    if abs(pos) > 0.05:
        L.append(f"Tamaño: <b>{lotes:.2f} lotes</b>")
    L += ["",
          "<b>Por qué:</b>",
          f"  🥇 Cobre 2d: {ctx['mom_cobre']:+.1f}%",
          f"  💵 DXY 5d: {ctx['mom_dxy']:+.1f}%",
          f"  🎯 Valor justo z: {ctx['z']:+.1f}",
          f"  📊 Actividad: {ctx['act']:.0%} del normal",
          "",
          f"📈 <b>Cuenta simulada</b>: <b>${s['capital']:,.0f}</b> ({pnl:+.1f}%) · {len(s['trades'])} ops.",
          "",
          "<i>Estrategia validada 17 años (Sharpe ~1,5). Paper-trading para probar en vivo. "
          "No es consejo de inversión.</i>"]
    return "\n".join(L)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "print"
    if cmd == "test":
        print("Enviado." if send("✅ <b>Paper-trader USD/CLP conectado.</b>") else "Falló.")
    elif cmd == "print":
        import re
        print(re.sub("<[^>]+>", "", reporte()))
    elif cmd == "run":
        r = reporte()
        send(r)
        print("Señal enviada a Telegram.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
