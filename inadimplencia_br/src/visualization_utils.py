import logging
import os
from io import BytesIO
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SAIDA

logger = logging.getLogger(__name__)


class PainelSaida:
    def __init__(
        self,
        base_dir,
        arquivo_painel,
        titulo_painel,
        gerar_individuais=None,
        miniatura_dpi=None,
        painel_dpi=None,
        colunas=2,
        painel_dir=None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # painel_dir permite salvar o consolidado em pasta diferente das individuais
        _dir_painel = Path(painel_dir) if painel_dir else self.base_dir
        _dir_painel.mkdir(parents=True, exist_ok=True)
        self.arquivo_painel = _dir_painel / arquivo_painel
        self.titulo_painel = titulo_painel
        self.gerar_individuais = (
            os.getenv("GERAR_IMAGENS_INDIVIDUAIS", "0") == "1"
            if gerar_individuais is None
            else gerar_individuais
        )
        self.miniatura_dpi = miniatura_dpi or SAIDA['dpi_miniatura']
        self.painel_dpi    = painel_dpi    or SAIDA['dpi_painel']
        self.colunas = colunas
        self.figuras = []

    def registrar(self, fig, nome_arquivo, titulo=None):
        if self.gerar_individuais:
            fig.savefig(self.base_dir / nome_arquivo, dpi=self.painel_dpi, bbox_inches="tight")
            logger.info(f"   [OK] Grafico '{nome_arquivo}' salvo!")

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=self.miniatura_dpi, bbox_inches="tight")
        buffer.seek(0)
        imagem = plt.imread(buffer)
        buffer.close()

        self.figuras.append({
            "titulo": titulo or Path(nome_arquivo).stem.replace("_", " ").title(),
            "imagem": imagem,
        })
        plt.close(fig)

    def finalizar(self):
        if not self.figuras:
            return None

        linhas = ceil(len(self.figuras) / self.colunas)
        fig, axes = plt.subplots(linhas, self.colunas, figsize=(18, max(6, linhas * 4.8)))
        axes = np.atleast_1d(axes).ravel()

        for ax, item in zip(axes, self.figuras):
            ax.imshow(item["imagem"])
            ax.set_title(item["titulo"], fontsize=11, fontweight="bold")
            ax.axis("off")

        for ax in axes[len(self.figuras):]:
            ax.remove()

        modo = "individual + painel" if self.gerar_individuais else "somente painel"
        fig.suptitle(self.titulo_painel, fontsize=20, fontweight="bold", y=0.995)
        fig.text(
            0.5, 0.005,
            f"Total de graficos: {len(self.figuras)} | Modo: {modo}",
            ha="center", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.98))
        fig.savefig(self.arquivo_painel, dpi=self.painel_dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"\n[OK] Painel consolidado salvo em '{self.arquivo_painel.name}'.")
        return self.arquivo_painel
