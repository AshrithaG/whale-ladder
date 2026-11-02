"""Create private launcher copies selecting WHALE's existing hybrid engine.

The installed WHALE checkout and its disaggregated entrypoint stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(domain, destination):
    launcher = domain/'scripts/train_chess_puzzle_multinode_disagg.sh'
    preamble = domain/'scripts/chess_puzzle_preamble_disagg_rsft.py'
    source = launcher.read_text()
    pre_source = preamble.read_text()
    call = 'python3 "$PROJECT_DIR/scripts/chess_puzzle_preamble_disagg_rsft.py"'
    import_line = 'from verl.trainer.main_textarena_disagg_rsft import main as _main'
    if source.count(call) != 1 or pre_source.count(import_line) != 1:
        raise ValueError('Upstream launcher changed; refusing an ambiguous replacement')
    destination.mkdir(parents=True, exist_ok=True)
    hybrid_pre = destination/'preamble_hybrid.py'
    hybrid_sh = destination/'train_hybrid.sh'
    hybrid_pre.write_text(pre_source.replace(import_line, 'from verl.trainer.main_ppo import main as _main'))
    hybrid_sh.write_text(source.replace(call, 'python3 "${LADDER_HYBRID_PREAMBLE:?}"'))
    (destination/'source.json').write_text(json.dumps({
        'launcher_source': str(launcher), 'launcher_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'preamble_source': str(preamble), 'preamble_sha256': hashlib.sha256(pre_source.encode()).hexdigest(),
        'entrypoint': 'verl.trainer.main_ppo', 'shared_gpu_pool': True}, indent=2)+'\n')
    return hybrid_sh, hybrid_pre


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('domain', type=Path)
    p.add_argument('destination', type=Path)
    a=p.parse_args()
    prepare(a.domain, a.destination)
