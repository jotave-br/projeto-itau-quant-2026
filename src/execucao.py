"""
Versionamento de execucoes: pasta da run e manifesto.

Substituto leve do MLflow: um projeto com uma duzia de configuracoes relevantes
nao precisa de framework de tracking. O que o MLflow guardaria - config,
artefatos, versoes, hashes - cabe num diretorio que da para abrir no explorador
de arquivos sem depender de ferramenta nenhuma.

Cada execucao cria uma pasta nova. Nada e sobrescrito.

    outputs/runs/2026-07-28_153000_v1/
    +-- config.json      parametros exatos daquela rodada
    +-- manifest.json    ambiente, commit, versoes, hashes dos dados, status
    +-- tabelas/         CSVs dos resultados
    +-- figuras/         PNGs
    +-- logs/            log da execucao

O manifesto e o que torna um numero auditavel: com ele da para dizer qual
codigo, quais dados e quais parametros produziram aquele resultado.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path

# Raiz do projeto: este arquivo esta em <raiz>/src/execucao.py
RAIZ = Path(__file__).resolve().parent.parent
DIR_RUNS = RAIZ / "outputs" / "runs"

# Bibliotecas cujas versoes sao registradas no manifesto.
_DEPENDENCIAS = ("numpy", "pandas", "scipy", "statsmodels", "yfinance", "matplotlib")


@dataclass(frozen=True)
class Execucao:
    """Uma execucao do pipeline, com sua pasta de saida ja criada."""

    id: str
    pasta: Path

    @property
    def tabelas(self) -> Path:
        return self.pasta / "tabelas"

    @property
    def figuras(self) -> Path:
        return self.pasta / "figuras"

    @property
    def logs(self) -> Path:
        return self.pasta / "logs"


def _commit_git() -> str | None:
    """Hash do commit atual, ou None se nao for repositorio git."""
    try:
        r = subprocess.run(
            ["git", "-C", str(RAIZ), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _arvore_suja() -> bool | None:
    """
    True se ha alteracoes nao commitadas.

    Resultado gerado com a arvore suja nao e reproduzivel pelo commit
    registrado, entao o manifesto precisa avisar.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(RAIZ), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versoes() -> dict[str, str]:
    """Versao instalada de cada dependencia relevante."""
    out = {}
    for nome in _DEPENDENCIAS:
        try:
            out[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            out[nome] = "nao instalado"
    return out


def hash_arquivo(caminho: Path, blocos: int = 1 << 20) -> str:
    """
    SHA-256 de um arquivo, lido em pedacos (os COTAHIST sao grandes).

    E como duas execucoes provam que usaram o mesmo dado de entrada sem
    precisar versionar o dado.
    """
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for pedaco in iter(lambda: f.read(blocos), b""):
            h.update(pedaco)
    return h.hexdigest()


def criar_execucao(rotulo: str = "v1") -> Execucao:
    """Cria a pasta da execucao com a estrutura padrao."""
    id_exec = f"{datetime.now():%Y-%m-%d_%H%M%S}_{rotulo}"
    pasta = DIR_RUNS / id_exec
    for sub in ("tabelas", "figuras", "logs"):
        (pasta / sub).mkdir(parents=True, exist_ok=True)
    return Execucao(id=id_exec, pasta=pasta)


def configurar_log(execucao: Execucao, nome: str = "pipeline") -> logging.Logger:
    """Log simultaneo em arquivo (dentro da run) e no terminal."""
    logger = logging.getLogger(nome)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    arquivo = logging.FileHandler(execucao.logs / f"{nome}.log", encoding="utf-8")
    arquivo.setFormatter(fmt)
    logger.addHandler(arquivo)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    return logger


def gravar_manifesto(
    execucao: Execucao,
    config: dict,
    arquivos_dados: list[Path] | None = None,
    status: str = "iniciada",
    extras: dict | None = None,
) -> Path:
    """
    Grava manifest.json e config.json na pasta da execucao.

    Chamar no inicio com status="iniciada" e no fim com "concluida" ou
    "falhou". Pasta que ficou em "iniciada" e execucao interrompida, e os
    numeros dela nao valem.
    """
    manifesto = {
        "id_execucao": execucao.id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "git": {
            "commit": _commit_git(),
            "arvore_com_alteracoes_nao_commitadas": _arvore_suja(),
        },
        "ambiente": {
            "python": sys.version.split()[0],
            "plataforma": platform.platform(),
            "dependencias": _versoes(),
        },
        "seed": config.get("seed"),
        "config": config,
        "dados_entrada": [
            {
                "arquivo": str(p.relative_to(RAIZ)) if RAIZ in p.parents else str(p),
                "sha256": hash_arquivo(p),
                "bytes": p.stat().st_size,
            }
            for p in (arquivos_dados or [])
            if p.exists()
        ],
    }
    if extras:
        manifesto["extras"] = extras

    (execucao.pasta / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    destino = execucao.pasta / "manifest.json"
    destino.write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return destino
