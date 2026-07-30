import os
import re

def patch_params(path, **params):
    """Reecrit dsCad.txt et dsLoad.txt avec de nouvelles valeurs de parametres."""
    for filename in ('dsCad.txt', 'dsLoad.txt'):
        fpath = os.path.join(path, filename)
        with open(fpath, 'r') as f:
            content = f.read()
        for name, value in params.items():
            content = re.sub(r'^' + name + r'\s*=.*$', f'{name}    = {value:.10f}', content, count=1, flags=re.MULTILINE)
        with open(fpath, 'w') as f:
            f.write(content)