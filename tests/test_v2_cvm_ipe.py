from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile

import pandas as pd
import pytest

from src.v2 import cvm_ipe


def _linha(**alteracoes) -> dict[str, str]:
    base = {
        "CNPJ_Companhia": "00.000.000/0001-91",
        "Nome_Companhia": "COMPANHIA AÇÚCAR S.A.",
        "Codigo_CVM": "001023",
        "Data_Referencia": "2026-01-09",
        "Categoria": "Fato Relevante",
        "Tipo": "",
        "Especie": "",
        "Assunto": "Distribuição de dividendos",
        "Data_Entrega": "2026-01-09",
        "Tipo_Apresentacao": "AP - Apresentação",
        "Protocolo_Entrega": "001023IPE090120260000000001-01",
        "Versao": "1",
        "Link_Download": "https://www.rad.cvm.gov.br/doc/1",
    }
    base.update(alteracoes)
    return base


def _zip_ipe(
    linhas: list[dict[str, str]],
    colunas: tuple[str, ...] = cvm_ipe.COLUNAS_IPE,
) -> bytes:
    texto = io.StringIO(newline="")
    escritor = csv.DictWriter(
        texto, fieldnames=colunas, delimiter=";", lineterminator="\n"
    )
    escritor.writeheader()
    escritor.writerows(linhas)
    saida = io.BytesIO()
    with zipfile.ZipFile(saida, "w", compression=zipfile.ZIP_DEFLATED) as arquivo:
        arquivo.writestr("ipe_cia_aberta_2026.csv", texto.getvalue().encode("cp1252"))
    return saida.getvalue()


def test_url_anual_oficial():
    assert cvm_ipe.url_ipe_ano(2026) == (
        "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/"
        "ipe_cia_aberta_2026.zip"
    )
    with pytest.raises(ValueError):
        cvm_ipe.url_ipe_ano(2002)


def test_le_zip_em_memoria_cp1252_e_por_path(tmp_path):
    bruto = _zip_ipe([_linha()])

    em_memoria = cvm_ipe.ler_zip_ipe(io.BytesIO(bruto))
    assert tuple(em_memoria.columns) == cvm_ipe.COLUNAS_IPE
    assert em_memoria.at[0, "Nome_Companhia"] == "COMPANHIA AÇÚCAR S.A."
    assert em_memoria.at[0, "Codigo_CVM"] == "001023"

    caminho = tmp_path / "ipe.zip"
    caminho.write_bytes(bruto)
    por_path = cvm_ipe.ler_zip_ipe(caminho)
    pd.testing.assert_frame_equal(em_memoria, por_path)


@pytest.mark.parametrize(
    "colunas",
    [
        cvm_ipe.COLUNAS_IPE[:-1],
        (*cvm_ipe.COLUNAS_IPE, "Coluna_Nova"),
        (cvm_ipe.COLUNAS_IPE[1], cvm_ipe.COLUNAS_IPE[0], *cvm_ipe.COLUNAS_IPE[2:]),
    ],
)
def test_schema_tem_13_colunas_exatas_e_na_ordem(colunas):
    linha = {chave: valor for chave, valor in _linha().items() if chave in colunas}
    if "Coluna_Nova" in colunas:
        linha["Coluna_Nova"] = "surpresa"
    with pytest.raises(cvm_ipe.ErroSchemaIPE, match="schema IPE invalido"):
        cvm_ipe.ler_zip_ipe(_zip_ipe([linha], colunas=colunas))


@pytest.mark.parametrize(
    "coluna,valor",
    [
        ("Data_Entrega", "2026-01-09 10:30:00"),
        ("Data_Entrega", "2026-02-30"),
        ("Data_Referencia", "09/01/2026"),
        ("Data_Referencia", ""),
    ],
)
def test_datas_exigem_yyyy_mm_dd_real_sem_hora(coluna, valor):
    with pytest.raises(cvm_ipe.ErroSchemaIPE, match=coluna):
        cvm_ipe.ler_zip_ipe(_zip_ipe([_linha(**{coluna: valor})]))


def test_data_referencia_fora_do_horizonte_e_preservada_sem_reparo():
    df = cvm_ipe.ler_zip_ipe(
        _zip_ipe([_linha(Data_Referencia="5019-07-26")])
    )

    assert df.at[0, "Data_Referencia"] == "5019-07-26"


def test_filtro_exato_preserva_reapresentacoes_e_ids_sao_deterministicos():
    linhas = [
        _linha(),
        _linha(
            Tipo_Apresentacao="RE - Reapresentação Espontânea",
            Protocolo_Entrega="001023IPE090120260000000002-02",
            Versao="2",
            Link_Download="https://www.rad.cvm.gov.br/doc/2",
        ),
        _linha(
            Categoria="Comunicado ao Mercado",
            Protocolo_Entrega="001023IPE090120260000000003-03",
        ),
        _linha(
            Categoria="fato relevante",
            Protocolo_Entrega="001023IPE090120260000000004-04",
        ),
    ]
    fatos = cvm_ipe.carregar_fatos_relevantes(_zip_ipe(linhas))

    assert fatos["Versao"].tolist() == ["1", "2"]
    assert fatos[cvm_ipe.COLUNA_ID].nunique() == 2

    invertidos = cvm_ipe.carregar_fatos_relevantes(_zip_ipe(list(reversed(linhas))))
    por_protocolo = fatos.set_index("Protocolo_Entrega")[cvm_ipe.COLUNA_ID].to_dict()
    por_protocolo_invertido = (
        invertidos.set_index("Protocolo_Entrega")[cvm_ipe.COLUNA_ID].to_dict()
    )
    assert por_protocolo == por_protocolo_invertido
    assert all(len(identificador) == 64 for identificador in por_protocolo.values())


def test_proxima_sessao_e_estritamente_posterior_via_calendario():
    calendario = pd.DatetimeIndex(["2026-01-09", "2026-01-12", "2026-01-13"])

    assert cvm_ipe.proxima_sessao_b3("2026-01-09", calendario) == pd.Timestamp(
        "2026-01-12"
    )
    assert cvm_ipe.proxima_sessao_b3("2026-01-10", calendario) == pd.Timestamp(
        "2026-01-12"
    )
    assert cvm_ipe.proxima_sessao_b3("2026-01-12", calendario) == pd.Timestamp(
        "2026-01-13"
    )

    with pytest.raises(ValueError, match="nao cobre"):
        cvm_ipe.proxima_sessao_b3("2026-01-13", calendario)


def test_adiciona_sessao_sem_agrupar_documentos():
    fatos = cvm_ipe.carregar_fatos_relevantes(
        _zip_ipe(
            [
                _linha(),
                _linha(
                    Data_Entrega="2026-01-10",
                    Versao="2",
                    Protocolo_Entrega="001023IPE100120260000000002-02",
                ),
            ]
        )
    )
    calendario = pd.DatetimeIndex(["2026-01-09", "2026-01-12", "2026-01-13"])

    saida = cvm_ipe.adicionar_sessao_disponivel(fatos, calendario)

    assert len(saida) == 2
    assert saida[cvm_ipe.COLUNA_SESSAO].tolist() == [
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-01-12"),
    ]


def test_download_zip_e_atomico_cacheado_com_hash(tmp_path):
    bruto = _zip_ipe([_linha()])
    chamadas: list[str] = []

    def cliente(url):
        chamadas.append(url)
        return cvm_ipe.RespostaHTTP(
            bruto, content_type="application/zip", url_final=url
        )

    primeiro = cvm_ipe.baixar_zip_ipe(2026, tmp_path, cliente_http=cliente)
    segundo = cvm_ipe.baixar_zip_ipe(2026, tmp_path, cliente_http=cliente)

    assert len(chamadas) == 1
    assert not primeiro.de_cache
    assert segundo.de_cache
    assert primeiro.sha256 == hashlib.sha256(bruto).hexdigest()
    assert primeiro.caminho.read_bytes() == bruto
    assert not list(tmp_path.glob("*.tmp"))
    meta = json.loads(
        primeiro.caminho.with_suffix(".zip.meta.json").read_text("utf-8")
    )
    assert meta["sha256"] == primeiro.sha256
    assert cvm_ipe.ler_zip_ipe(primeiro.caminho).shape == (1, 13)


def test_download_zip_invalido_nao_entra_no_cache(tmp_path):
    def cliente(_url):
        return b"<html>erro</html>"

    with pytest.raises(cvm_ipe.ErroSchemaIPE):
        cvm_ipe.baixar_zip_ipe(2026, tmp_path, cliente_http=cliente)
    assert not any(tmp_path.iterdir())


def test_documento_usa_magic_pdf_mesmo_com_mime_errado_e_cacheia(tmp_path):
    pdf = b"%PDF-1.7\nconteudo sintetico\n%%EOF"
    chamadas = 0

    def cliente(_url):
        nonlocal chamadas
        chamadas += 1
        return cvm_ipe.RespostaHTTP(pdf, content_type="text/html")

    destino = tmp_path / "documento.pdf"
    primeiro = cvm_ipe.baixar_documento_ipe(
        "https://www.rad.cvm.gov.br/doc/1",
        destino,
        cliente_http=cliente,
    )
    segundo = cvm_ipe.baixar_documento_ipe(
        "https://www.rad.cvm.gov.br/doc/1",
        destino,
        cliente_http=cliente,
    )

    assert chamadas == 1
    assert primeiro.eh_pdf is True
    assert primeiro.content_type == "text/html"
    assert primeiro.sha256 == hashlib.sha256(pdf).hexdigest()
    assert segundo.de_cache


def test_mime_pdf_nao_salva_conteudo_sem_assinatura(tmp_path):
    def cliente(_url):
        return cvm_ipe.RespostaHTTP(
            b"<html>login</html>", content_type="application/pdf"
        )

    destino = tmp_path / "documento.pdf"
    with pytest.raises(cvm_ipe.ErroDocumentoIPE, match="%PDF-"):
        cvm_ipe.baixar_documento_ipe(
            "https://www.rad.cvm.gov.br/doc/1",
            destino,
            cliente_http=cliente,
        )
    assert not destino.exists()
    assert not destino.with_suffix(".pdf.meta.json").exists()
