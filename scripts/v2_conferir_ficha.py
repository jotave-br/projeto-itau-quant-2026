"""Confere uma ficha preenchida contra a original antes de avaliar.

Existe para pegar estrago de planilha (encoding, separador, texto alterado)
enquanto ainda dá para refazer, e não depois de horas de rotulagem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.v2_05_avaliar_validacao_humana import (  # noqa: E402
    _ler_csv,
    normalizar_ficha,
)


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ficha", type=Path)
    parser.add_argument("original", type=Path)
    return parser.parse_args(argv)


def _separador_errado(caminho: Path) -> bool:
    """Planilha em locale pt-BR tende a salvar CSV com ponto e vírgula."""
    try:
        with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
            cabecalho = arquivo.readline()
    except (OSError, UnicodeDecodeError):
        return False
    return ";" in cabecalho and "," not in cabecalho


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    if _separador_errado(args.ficha.resolve()):
        print("A ficha foi salva com ponto e virgula como separador; o pipeline")
        print("espera virgula. No LibreOffice, marque virgula ao exportar.")
        return 1
    try:
        ficha_bruta = _ler_csv(args.ficha.resolve())
    except UnicodeDecodeError:
        print("A ficha nao esta em UTF-8. A planilha salvou no codepage do")
        print("Windows e os acentos foram destruidos.")
        print("Reabra a original e salve de novo escolhendo UTF-8.")
        return 1
    except ValueError as exc:
        print(f"Nao deu para ler a ficha: {exc}")
        return 1
    original = _ler_csv(args.original.resolve())

    if len(ficha_bruta.columns) == 1:
        print("A ficha veio com uma coluna so. A planilha salvou usando ponto e")
        print("virgula como separador; o pipeline espera virgula.")
        return 1

    problemas: list[str] = []

    if len(ficha_bruta) != len(original):
        problemas.append(
            f"a ficha tem {len(ficha_bruta)} linhas e a original tem {len(original)}"
        )

    ids_ficha = ficha_bruta.get("id_anonimo")
    ids_orig = original["id_anonimo"]
    if ids_ficha is None:
        problemas.append("coluna id_anonimo sumiu da ficha")
    elif set(ids_ficha) != set(ids_orig):
        faltando = sorted(set(ids_orig) - set(ids_ficha))[:5]
        sobrando = sorted(set(ids_ficha) - set(ids_orig))[:5]
        problemas.append(f"IDs mudaram; faltando={faltando} sobrando={sobrando}")
    else:
        base = original.set_index("id_anonimo")["texto"]
        atual = ficha_bruta.set_index("id_anonimo")["texto"]
        alterados = [
            codigo for codigo in base.index if atual.at[codigo] != base.at[codigo]
        ]
        if alterados:
            problemas.append(
                f"{len(alterados)} texto(s) alterados pela planilha; "
                f"primeiros={alterados[:5]}"
            )

    preenchidas = 0
    if "especifico_empresa" in ficha_bruta and "direcao" in ficha_bruta:
        vazio = ficha_bruta["especifico_empresa"].astype(str).str.strip().eq("")
        preenchidas = int((~vazio).sum())
    else:
        problemas.append("colunas de resposta sumiram da ficha")

    completa = preenchidas == len(original)
    if completa:
        # Só vale rodar o normalizador quando não há linha em branco.
        try:
            normalizar_ficha(ficha_bruta, "ficha")
        except ValueError as exc:
            problemas.append(f"o v2_05 recusaria: {exc}")

    print(f"linhas: {len(ficha_bruta)}")
    print(f"respostas preenchidas: {preenchidas}/{len(original)}")
    if problemas:
        print("\nPROBLEMAS:")
        for item in problemas:
            print(f"  - {item}")
        print("\nNao distribua nem continue: corrija a ferramenta ou os ajustes.")
        return 1
    if not completa:
        print("\nTexto e IDs intactos. Ficha ainda incompleta, o que e esperado")
        print("em teste de ida e volta da planilha.")
        return 0
    print("\nTexto e IDs intactos e ficha completa. Pronta para o v2_05.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
