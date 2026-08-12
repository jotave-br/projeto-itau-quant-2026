from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import v2_02_preparar_documentos as preparar
from src.v2 import cvm_ipe, documentos


def _par() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "janela": ["2026-01"],
            "lider": ["LID3"],
            "seguidora": ["SEG3"],
            "emissor_lider": ["LID"],
            "emissor_seguidora": ["SEG"],
            "faixa_minima": [20],
        }
    )


def _fatos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID_Documento": ["aceito", "reapresentado", "nao-lider", "fora"],
            "Tipo_Apresentacao": [
                "AP - Apresentação",
                "RE - Reapresentação Espontânea",
                "AP - Apresentação",
                "AP - Apresentação",
            ],
            "Versao": ["1", "2", "1", "1"],
            "Data_Entrega": [
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
                "2025-12-29",
            ],
            "emissor_id": ["LID", "LID", "OUTRO", "LID"],
            "Link_Download": ["https://cvm.invalid/documento"] * 4,
        }
    )


def test_janelas_sao_derivadas_dos_rotulos_congelados_da_v1():
    pares = pd.concat(
        [_par(), _par().assign(janela="2026-04")], ignore_index=True
    )

    janelas = preparar._janelas_dos_pares(pares)

    assert [janela.rotulo for janela in janelas] == ["2026-01", "2026-04"]
    assert janelas[0].teste_inicio == pd.Timestamp("2026-01-01")
    assert janelas[0].teste_fim == pd.Timestamp("2026-04-01")
    assert janelas[0].treino_inicio == pd.Timestamp("2024-01-01")


@pytest.mark.parametrize("rotulo", ["", "2026", "2026-13", "abril-2026"])
def test_rotulo_de_janela_invalido_e_recusado(rotulo):
    with pytest.raises(ValueError, match="inválido|sem janelas"):
        preparar._janelas_dos_pares(_par().assign(janela=rotulo))


def test_funil_mantem_so_ap_de_lider_na_janela_e_diagnostica_o_resto():
    calendario = pd.DatetimeIndex(
        ["2025-12-30", "2026-01-02", "2026-01-05", "2026-01-06"]
    )

    selecionados, diagnosticos, etapas = preparar.selecionar_corpus(
        _fatos(), _par(), calendario
    )

    assert selecionados["ID_Documento"].tolist() == ["aceito"]
    motivos = diagnosticos.set_index("ID_Documento")["motivo_diagnostico"]
    assert motivos["reapresentado"] == "reapresentacao_ou_versao_nao_original"
    assert motivos["nao-lider"] == "emissor_nao_e_lider_top20_na_janela"
    assert motivos["fora"] == "fora_das_janelas_de_teste"
    assert etapas[-1] == {"etapa": "lider_top20_na_janela", "documentos": 1}
    assert selecionados.loc[0, "Sessao_Disponivel"] == pd.Timestamp("2026-01-05")


def test_processamento_registra_extracao_e_limite_do_texto(monkeypatch, tmp_path):
    pdf = tmp_path / "documento.pdf"
    pdf.write_bytes(b"%PDF-falso-para-o-mock")
    artefato = cvm_ipe.ArtefatoBaixado(
        caminho=pdf,
        url="https://cvm.invalid/documento",
        sha256="a" * 64,
        tamanho=23,
        content_type="application/pdf",
        de_cache=True,
        eh_pdf=True,
    )
    extracao = documentos.ResultadoExtracao(
        sha256="a" * 64,
        paginas_total=2,
        paginas_com_texto=2,
        texto="informação relevante " * 30,
        caracteres=len("informação relevante " * 30),
        status=documentos.STATUS_OK,
    )
    monkeypatch.setattr(cvm_ipe, "baixar_documento_ipe", lambda *a, **k: artefato)
    monkeypatch.setattr(documentos, "extrair_texto_pdf", lambda _p: extracao)

    resultado = preparar._processar_documento(
        {
            "ID_Documento": "doc-1",
            "Link_Download": "https://cvm.invalid/documento",
        },
        forcar=False,
    )

    assert resultado["status_documento"] == "ok"
    assert resultado["pdf_sha256"] == "a" * 64
    assert resultado["paginas_total"] == 2
    assert resultado["texto_llm"] == extracao.texto.strip()
    assert resultado["pdf_cache"] is True


def test_processamento_isola_falha_de_um_documento(monkeypatch):
    def falhar(*_args, **_kwargs):
        raise cvm_ipe.ErroDownloadIPE("indisponível")

    monkeypatch.setattr(cvm_ipe, "baixar_documento_ipe", falhar)
    resultado = preparar._processar_documento(
        {
            "ID_Documento": "doc-erro",
            "Link_Download": "https://cvm.invalid/erro",
        },
        forcar=False,
    )

    assert resultado["status_documento"] == "erro"
    assert "ErroDownloadIPE" in resultado["erro_documento"]
    assert resultado["texto"] == ""
