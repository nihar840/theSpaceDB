"""
shell.py — spacesh: interactive CLI shell for theSpaceDB

Like mongosh but for a cognitive database.

Usage:
    python -m spacedb.shell D:/SpaceDB/data
    spacesh D:/SpaceDB/data          # if installed via pip

Commands:
    use <name>              Switch to a space
    ingest <text>           Store a memory
    query <text>            Query nearest memories (default: 100ms budget)
    query <text> --time N   Query with N ms time budget (deeper = N higher)
    reinforce <id1> <id2>   Pull two blocks closer
    decay <id1> <id2>       Push two blocks apart
    clusters                List all clusters
    personalities           List personality clusters only
    drift on [N]            Start drift (optional idle threshold in seconds)
    drift off               Stop drift
    drift status            Show drift status
    status                  Show space status
    spaces                  List all spaces in this client
    help                    Show this help
    exit / quit             Exit the shell
"""

import sys
import os
import time
import shlex
import textwrap
from typing import Optional


# ── ANSI colours ─────────────────────────────────────────────────────────────
C = {
    'cyan':    '\033[96m',
    'yellow':  '\033[93m',
    'green':   '\033[92m',
    'red':     '\033[91m',
    'grey':    '\033[90m',
    'white':   '\033[97m',
    'bold':    '\033[1m',
    'reset':   '\033[0m',
    'magenta': '\033[95m',
    'blue':    '\033[94m',
}

def c(color: str, text: str) -> str:
    return f"{C.get(color,'')}{text}{C['reset']}"


BANNER = f"""
{C['cyan']}
  ╔══════════════════════════════════════════════════╗
  ║                                                  ║
  ║   ∞  theSpaceDB Shell  v0.1.0                    ║
  ║      Blocks in free space. Infinity begins here. ║
  ║                                                  ║
  ╚══════════════════════════════════════════════════╝
{C['reset']}  Type {c('yellow','help')} to see available commands.
"""

HELP_TEXT = f"""
{c('cyan','  COMMANDS')}
  {c('yellow','use')} <name>               Switch to / create a space
  {c('yellow','ingest')} <text>            Store a memory block
  {c('yellow','query')} <text>             Query (default 100ms budget)
  {c('yellow','query')} <text> --time N    Query with N ms time budget
  {c('yellow','reinforce')} <id1> <id2>    Pull two blocks closer
  {c('yellow','decay')} <id1> <id2>        Push two blocks apart
  {c('yellow','clusters')}                 List all clusters
  {c('yellow','personalities')}            List personality clusters
  {c('yellow','drift')} on [N]             Start drift (idle threshold N sec)
  {c('yellow','drift')} off               Stop drift
  {c('yellow','drift')} status            Show drift info
  {c('yellow','status')}                  Space status
  {c('yellow','spaces')}                  List all spaces
  {c('yellow','help')}                    Show this help
  {c('yellow','exit')}                    Exit

{c('grey','  TIP: query with higher --time for deeper memory exploration.')}
"""


# ── Shell class ───────────────────────────────────────────────────────────────
class SpaceShell:

    def __init__(self, root: str):
        self._root   = root
        self._client = None
        self._space  = None
        self._name   = None
        self._init_client(root)

    def _init_client(self, root: str):
        from .client import SpaceClient
        self._client = SpaceClient(root, silent=True)

    def _prompt(self) -> str:
        if self._name:
            return f"{c('cyan','space')}[{c('yellow', self._name)}]> "
        return f"{c('cyan','space')}> "

    # ── main loop ─────────────────────────────────────────────
    def run(self):
        print(BANNER)
        print(c('grey', f"  Connected: {self._root}\n"))
        while True:
            try:
                raw = input(self._prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{c('grey','  Bye. The space remembers you.')}")
                break
            if not raw:
                continue
            try:
                self._dispatch(raw)
            except Exception as e:
                print(f"  {c('red', 'Error:')} {e}")

    def _dispatch(self, raw: str):
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = raw.split()
        if not parts:
            return
        cmd, args = parts[0].lower(), parts[1:]

        if   cmd in ('exit', 'quit'):       raise SystemExit
        elif cmd == 'help':                  print(HELP_TEXT)
        elif cmd == 'use':                   self._cmd_use(args)
        elif cmd == 'spaces':                self._cmd_spaces()
        elif cmd == 'ingest':                self._cmd_ingest(args)
        elif cmd == 'query':                 self._cmd_query(args)
        elif cmd == 'reinforce':             self._cmd_reinforce(args)
        elif cmd == 'decay':                 self._cmd_decay(args)
        elif cmd == 'clusters':              self._cmd_clusters()
        elif cmd == 'personalities':         self._cmd_personalities()
        elif cmd == 'drift':                 self._cmd_drift(args)
        elif cmd == 'status':                self._cmd_status()
        else:
            print(f"  {c('red','Unknown command:')} {cmd}  "
                  f"(type {c('yellow','help')})")

    # ── commands ──────────────────────────────────────────────
    def _cmd_use(self, args):
        if not args:
            print(f"  {c('red','Usage:')} use <space_name>"); return
        name = args[0]
        self._space = self._client.use(name)
        self._name  = name
        s = self._space.status()
        print(f"  {c('green','↳')} Switched to space: {c('yellow', name)}")
        print(f"  {c('grey', f'  blocks={s[\"blocks\"]}  clusters={s[\"clusters\"]}  personalities={s[\"personalities\"]}')}")

    def _cmd_spaces(self):
        spaces = self._client.list_spaces()
        if not spaces:
            print(f"  {c('grey','No spaces yet. Use: use <name>')}"); return
        print(f"\n  {c('cyan','Spaces')} ({len(spaces)})")
        for s in spaces:
            print(f"    {c('yellow','·')} {s}")
        print()

    def _cmd_ingest(self, args):
        self._require_space()
        text = ' '.join(args)
        if not text:
            print(f"  {c('red','Usage:')} ingest <text>"); return
        t0    = time.time()
        block = self._space.ingest(text)
        ms    = (time.time() - t0) * 1000
        print(f"  {c('green','✦')} Block stored  "
              f"{c('grey', '[' + block.id[:8] + ']')}  "
              f"{c('grey', f'{ms:.1f}ms')}")
        print(f"  {c('grey','  ' + repr(text[:60]))}")

    def _cmd_query(self, args):
        self._require_space()
        # parse --time flag
        time_ms = 100
        text_parts = []
        i = 0
        while i < len(args):
            if args[i] == '--time' and i + 1 < len(args):
                time_ms = int(args[i + 1]); i += 2
            else:
                text_parts.append(args[i]); i += 1
        text = ' '.join(text_parts)
        if not text:
            print(f"  {c('red','Usage:')} query <text> [--time N]"); return

        t0      = time.time()
        results = self._space.query(text).within(ms=time_ms).fetch()
        ms      = (time.time() - t0) * 1000
        radius  = self._space._engine._radius(time_ms)

        self._print_query_results(text, results, time_ms, radius, ms)

    def _print_query_results(self, text, results, time_ms, radius, elapsed_ms):
        w = 58
        header  = f"Query: {text!r}  radius={radius:.2f}  budget={time_ms}ms"
        header  = header[:w].ljust(w)
        print(f"\n{c('cyan','  ╔' + '═'*w + '╗')}")
        print(f"{c('cyan','  ║')} {c('white', header)} {c('cyan','║')}")
        print(f"{c('cyan','  ╠' + '═'*w + '╣')}")
        if not results:
            blank = 'No results found.'.ljust(w)
            print(f"{c('cyan','  ║')} {c('grey', blank)} {c('cyan','║')}")
        for r in results:
            token   = r['token'][:38].ljust(38)
            score   = f"score: {r['score']:.4f}"
            cid     = f"[{r['cluster'][:6]}]" if r['cluster'] else '       '
            row     = f"{c('grey', r['id'][:8])}  {token}  {c('green', score)}  {c('grey', cid)}"
            # strip ANSI for padding, then reprint
            print(f"  {c('cyan','║')} {row}")
        print(f"{c('cyan','  ╚' + '═'*w + '╝')}")
        print(f"  {c('grey', f'{len(results)} results in {elapsed_ms:.1f}ms')}\n")

    def _cmd_reinforce(self, args):
        self._require_space()
        if len(args) < 2:
            print(f"  {c('red','Usage:')} reinforce <id1> <id2>"); return
        self._space.reinforce(args[0], args[1])
        print(f"  {c('green','↑')} Reinforced: {c('grey', args[0][:8])} ↔ {c('grey', args[1][:8])}")

    def _cmd_decay(self, args):
        self._require_space()
        if len(args) < 2:
            print(f"  {c('red','Usage:')} decay <id1> <id2>"); return
        self._space.decay(args[0], args[1])
        print(f"  {c('grey','↓')} Decayed: {args[0][:8]} ↔ {args[1][:8]}")

    def _cmd_clusters(self):
        self._require_space()
        clusters = self._space.clusters.all()
        if not clusters:
            print(f"  {c('grey','No clusters yet. Ingest more data.')}"); return
        print(f"\n  {c('cyan','Clusters')} ({len(clusters)})")
        print(f"  {'ID':<10} {'Name':<16} {'Blocks':>6}  {'Spirit':>8}  {'Type':<12}")
        print(f"  {'─'*10} {'─'*16} {'─'*6}  {'─'*8}  {'─'*12}")
        for cl in sorted(clusters, key=lambda x: -x['spirit_size']):
            tag  = c('yellow', 'PERSONALITY') if cl['personality'] else c('grey', 'cluster')
            name = (cl['name'] or '—')[:16]
            print(f"  {c('grey', cl['id']):<10} {name:<16} "
                  f"{cl['blocks']:>6}  {cl['spirit_size']:>8.3f}  {tag}")
        print()

    def _cmd_personalities(self):
        self._require_space()
        ps = self._space.clusters.personalities()
        if not ps:
            print(f"  {c('grey','No personalities yet. Spirit size needs to grow.')}"); return
        print(f"\n  {c('yellow','✦ Personalities')} ({len(ps)})")
        for p in ps:
            print(f"    {c('yellow','·')} {c('white', p['name'] or p['id'])}  "
                  f"blocks={p['blocks']}  spirit={p['spirit_size']:.3f}")
        print()

    def _cmd_drift(self, args):
        self._require_space()
        sub = args[0].lower() if args else ''
        if sub == 'on':
            idle = float(args[1]) if len(args) > 1 else 10.0
            self._space.drift.start(idle_seconds=idle)
        elif sub == 'off':
            self._space.drift.stop()
        elif sub == 'status':
            ds = self._space.drift.status()
            state = c('green', 'ON') if ds['active'] else c('grey', 'OFF')
            print(f"  Drift: {state}  "
                  f"idle={ds['idle_s']}s  "
                  f"input_rate={ds['input_rate']:.4f}/s")
        else:
            print(f"  {c('red','Usage:')} drift [on [N] | off | status]")

    def _cmd_status(self):
        self._require_space()
        s = self._space.status()
        drift = c('green', 'ON') if s['drift'] else c('grey', 'OFF')
        print(f"""
  {c('cyan', '── Space Status ─────────────────────────────')}
  Space            {c('yellow', s['space'])}
  Blocks           {c('white', str(s['blocks']))}
  Graph nodes      {s['graph_nodes']}  /  edges {s['graph_edges']}
  Clusters         {s['clusters']}
  Personalities    {c('yellow', str(s['personalities']))}
  P-Threshold      {s['threshold']}
  W updates        {s['w_updates']}
  Input rate       {s['input_rate']:.4f} /s
  Drift            {drift}
  Idle             {s['idle_s']}s
  {c('cyan', '─────────────────────────────────────────────')}
""")

    def _require_space(self):
        if not self._space:
            raise RuntimeError(
                f"No space selected. Use: {c('yellow','use <name>')}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else './spacedb_data'
    SpaceShell(path).run()


if __name__ == '__main__':
    main()
