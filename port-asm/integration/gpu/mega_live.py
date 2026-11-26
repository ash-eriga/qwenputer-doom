#!/usr/bin/env python3
"""Play ray0 live on the megakernel: the model alone runs the machine, the browser types the keys.

One process: a game thread runs the megakernel in launches of --chunk tokens and keeps the latest
frame (read out of the model's GDN state) as a PNG; an HTTP server shows it and takes the keys.
The only thing the host ever puts into the model is a typed byte over the UART (a KEY0 token at a
record boundary): the key mask (bit 0 up, 1 down, 2 left, 3 right) when you play, 0x80 to hand
control back to ray0's autopilot. No autopilot runs on the host and no memory is poked. lx.ref
checks every token beside the model (mega_play.Shadow; --no-check turns it off) and the page
shows whether the run is still verified.

    python port-asm/integration/gpu/mega_live.py --packed /path/packed.pt --port 8950
    cloudflared tunnel --url http://localhost:8950      # then open the printed URL

  GET  /            the page (mega_live.html)
  GET  /frame.png   the latest frame
  GET  /api/status  frame, tokens, tok/s, fps, verified, mode, mask
  POST /api/keys    {"mask": N}             (manual mode: typed when it changes)
  POST /api/mode    {"mode": "demo"|"manual"}

With --token SECRET the POST endpoints need the header X-Token: SECRET (open the page as
/?token=SECRET). The server binds 127.0.0.1 unless --host says otherwise: put it behind a
tunnel (Cloudflare Access for anything public) rather than exposing the port.
"""
from __future__ import annotations

import argparse
import hmac
import json
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "megakernel"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from packed_model import (add_lxsrc, build_decode_plan, load_packed_runtime,  # noqa: E402
                          make_session)

PAGE = HERE / "mega_live.html"


class Live:
    """The game thread and the state the HTTP handlers read and set."""

    def __init__(self, game, chunk: int, scale: int, max_tokens: int = 0):
        self.game, self.chunk, self.scale, self.max_tokens = game, chunk, scale, max_tokens
        self.lock = threading.Lock()
        self.mode, self.mask = "demo", 0
        # bytes to type, in order: a key tapped within one launch is still seen for a frame
        self.queue = deque(maxlen=8)
        self.png = game.png(scale)
        self.frame_times = deque(maxlen=64)
        self.tok_times = deque(maxlen=64)  # (time, tokens)
        self.error = None
        self.stop = threading.Event()
        self.started = time.time()

    def _queue(self, b: int) -> None:
        last = self.queue[-1] if self.queue else self.game.want_byte
        if b != last:
            self.queue.append(b)

    def set_keys(self, mask: int) -> None:
        with self.lock:
            self.mask = mask & 0x0F
            if self.mode == "manual":
                self._queue(self.mask)

    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.mode = mode
            if mode == "demo":
                self.mask = 0
                self._queue(0x80)
            else:
                self._queue(self.mask)

    def loop(self) -> None:
        g = self.game
        try:
            while not self.stop.is_set():
                if self.max_tokens and g.tokens >= self.max_tokens:
                    break
                with self.lock:
                    # the next byte once the last one is typed (the guest reads one a frame)
                    if self.queue and (g.want_byte is None or g.typed_byte == g.want_byte):
                        g.request(self.queue.popleft())
                new = g.step(self.chunk)
                now = time.time()
                self.tok_times.append((now, g.tokens))
                if new:
                    png = g.png(self.scale)
                    with self.lock:
                        self.png = png
                    self.frame_times.append((now, g.frame))
                if g.shadow is not None and not g.shadow.ok:
                    self.error = {"diverged": g.shadow.divergence}
                    print(json.dumps(self.error), flush=True)
                    break
        except Exception:
            self.error = {"exception": traceback.format_exc()}
            print(self.error["exception"], flush=True)

    @staticmethod
    def _rate(q) -> float:
        if len(q) < 2 or q[-1][0] <= q[0][0]:
            return 0.0
        return (q[-1][1] - q[0][1]) / (q[-1][0] - q[0][0])

    def status(self) -> dict:
        s = self.game.status()
        with self.lock:
            s.update(mode=self.mode, mask=self.mask, queued=list(self.queue))
        s.update(tok_per_s=round(self._rate(self.tok_times), 1),
                 fps=round(self._rate(self.frame_times), 2),
                 running=not self.stop.is_set() and self.error is None,
                 error=self.error, uptime_s=round(time.time() - self.started, 1),
                 width=self.game.view.width, height=self.game.view.height)
        return s


def make_handler(live: Live, token: str | None):
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/frame.png":
                with live.lock:
                    png = live.png
                self._send(200, png, "image/png")
            elif path == "/api/status":
                self._json(live.status())
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):
            if token is not None and not hmac.compare_digest(
                    self.headers.get("X-Token", ""), token):
                return self._json({"ok": False, "error": "bad token"}, 403)
            n = min(int(self.headers.get("Content-Length") or 0), 4096)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            path = self.path.split("?", 1)[0]
            if path == "/api/keys":
                live.set_keys(int(data.get("mask", 0)))
                self._json({"ok": True})
            elif path == "/api/mode":
                live.set_mode("manual" if data.get("mode") == "manual" else "demo")
                self._json({"ok": True, "mode": live.mode})
            else:
                self._json({"ok": False, "error": "not found"}, 404)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elf", default=str(HERE.parents[2] / "build" / "asm-ray0-small.elf"))
    ap.add_argument("--packed", required=True)
    ap.add_argument("--lxsrc", default=str(HERE.parents[3] / "Doom-On-Qwen" / "lxsrc"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu: gpu/cpu_decode.py's decoder in place of the kernel (~10 tok/s)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8950)
    ap.add_argument("--token", default=None, help="required as X-Token on POSTs")
    ap.add_argument("--chunk", type=int, default=512,
                    help="tokens a launch: smaller answers keys sooner, larger runs faster")
    ap.add_argument("--scale", type=int, default=1, help="PNG pixel scale (the page scales too)")
    ap.add_argument("--no-check", action="store_true", help="no lx.ref beside the model")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=0, help="stop the game after this many")
    ap.add_argument("--wad", default=None, help="doom1.wad for PLAYPAL colours")
    a = ap.parse_args(argv)

    add_lxsrc(a.lxsrc)
    import torch

    from asm_readout import read_playpal, write_palette_wad
    from mega_play import CpuMega, MegaGame

    torch.set_grad_enabled(False)
    rt = load_packed_runtime(a.elf, a.packed, a.lxsrc, device=a.device, compile_fast=False)
    rt.fast.dec = build_decode_plan(rt.fast, rt.model)
    sess = make_session(rt, 1)
    if a.device == "cuda":
        import mega

        t0 = time.time()
        mg = mega.Mega(rt.fast, rt.model, sess.cache, 0, spec=not a.no_spec)
        print(json.dumps({"mega_build_s": round(time.time() - t0, 1), **mg.stats}), flush=True)
    else:
        mg = CpuMega(rt, sess.cache)
    if a.wad:
        pal = read_playpal(a.wad)
    else:
        import tempfile

        pal = read_playpal(write_palette_wad(Path(tempfile.mkdtemp()) / "palette.wad"))
    game = MegaGame(rt, sess, mg, check=not a.no_check, palette=pal)
    live = Live(game, a.chunk, a.scale, a.max_tokens)
    th = threading.Thread(target=live.loop, daemon=True)
    th.start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(live, a.token))
    print(f"serving http://{a.host}:{a.port}/" + (f"?token={a.token}" if a.token else ""),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.stop.set()
        th.join(timeout=30)
        print(json.dumps(live.status()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
