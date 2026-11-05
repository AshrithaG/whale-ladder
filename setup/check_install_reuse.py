"""Prove a prepared installation's inputs match the selected runtime."""
import argparse
import json
from pathlib import Path
import subprocess

INPUTS=['setup/install.sh','setup/constraints','patches','shim','ladder/config.env']


def check(code,prepared,runtime):
    changed=subprocess.check_output(['git','diff','--name-only',prepared,runtime,'--',*INPUTS],cwd=code,text=True).splitlines()
    if changed:raise RuntimeError('Prepared installation inputs changed: '+', '.join(changed))
    return dict(installed_base_revision=prepared,runtime_revision=runtime,
        installation_inputs_checked=INPUTS,changed_installation_inputs=[],
        reused_environment=True,note='Installed-revision marker retained; runtime source is recorded separately.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',type=Path,required=True);p.add_argument('--prepared',required=True)
    p.add_argument('--runtime',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=check(a.code,a.prepared,a.runtime)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
