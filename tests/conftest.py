"""
Configuracao compartilhada dos testes.

Coloca a raiz do projeto no sys.path para que `import src.<modulo>` funcione
sem precisar instalar o pacote.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
