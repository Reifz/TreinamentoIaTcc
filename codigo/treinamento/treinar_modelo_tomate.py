"""Treinador principal do Modelo B MobileNetV2 para folhas de tomateiro.

O comando valida a estrutura completa do dataset, registra a execucao e usa
as rotinas de modelo, treinamento, metricas e graficos do modulo de dataset.
"""

from __future__ import annotations

import argparse
import atexit
import json
import platform
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

import treinar_modelo_tomate_dataset as base


RAIZ_PROJETO = Path(__file__).resolve().parents[2]
DATASET_PADRAO = RAIZ_PROJETO / "dataset"
RAIZ_SAIDAS = RAIZ_PROJETO / "execucoes" / "novas"
PREFIXO_EXECUCAO = "modelo_b_mobilenetv2"


class FluxoDuplicado:
    """Replica a saida no terminal e no arquivo de log da execucao."""

    def __init__(self, terminal, arquivo_log):
        self.terminal = terminal
        self.arquivo_log = arquivo_log

    def write(self, texto):
        self.terminal.write(texto)
        self.arquivo_log.write(texto)
        self.arquivo_log.flush()
        return len(texto)

    def flush(self):
        self.terminal.flush()
        self.arquivo_log.flush()

    def isatty(self):
        return self.terminal.isatty()


def iniciar_captura_console(diretorio_saida: Path) -> None:
    """Registra stdout e stderr no log da execucao."""
    arquivo_log = (diretorio_saida / "logs" / "console.log").open(
        "a", encoding="utf-8", buffering=1
    )
    saida_original = sys.stdout
    erro_original = sys.stderr
    sys.stdout = FluxoDuplicado(saida_original, arquivo_log)
    sys.stderr = FluxoDuplicado(erro_original, arquivo_log)

    def restaurar_fluxos():
        sys.stdout = saida_original
        sys.stderr = erro_original
        arquivo_log.close()

    atexit.register(restaurar_fluxos)


def analisar_argumentos() -> argparse.Namespace:
    """Le os argumentos de linha de comando do treinador principal."""
    analisador = argparse.ArgumentParser(
        description="Treina o Modelo B MobileNetV2 em um dataset de tomateiros."
    )
    analisador.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PADRAO,
        help="Caminho do dataset (padrao: dataset/).",
    )
    analisador.add_argument(
        "--batch", type=int, default=base.TAMANHO_LOTE_PADRAO
    )
    analisador.add_argument(
        "--epochs-head", type=int, default=base.EPOCAS_CABECALHO
    )
    analisador.add_argument(
        "--epochs-fine1", type=int, default=base.EPOCAS_AJUSTE_FINO_1
    )
    analisador.add_argument(
        "--epochs-fine2", type=int, default=base.EPOCAS_AJUSTE_FINO_2
    )
    analisador.add_argument(
        "--check-only",
        action="store_true",
        help="Valida estrutura, classes e contagens sem iniciar o treinamento.",
    )
    analisador.add_argument(
        "--require-gpu",
        action="store_true",
        help="Interrompe se nenhuma GPU estiver disponivel.",
    )
    analisador.add_argument(
        "--output-dir",
        type=Path,
        help="Diretorio exato da execucao; o padrao e execucoes/novas/<data-hora>.",
    )
    analisador.add_argument(
        "--export-h5",
        action="store_true",
        help="Exporta tambem H5 legado; o formato oficial continua sendo .keras.",
    )
    return analisador.parse_args()


def criar_identificador_seguro(valor: str) -> str:
    """Produz um identificador seguro para diretorios e modelos."""
    normalizado = re.sub(r"[^a-zA-Z0-9_-]+", "_", valor.strip())
    return normalizado.strip("_") or "dataset_personalizado"


def resolver_dataset(argumentos: argparse.Namespace) -> tuple[str, Path]:
    """Resolve o identificador e o caminho do dataset informado."""
    raiz = argumentos.dataset.expanduser()
    return criar_identificador_seguro(raiz.name), raiz


def listar_diretorios_classes(diretorio_conjunto: Path) -> set[str]:
    """Lista as pastas de classe de um conjunto obrigatorio."""
    if not diretorio_conjunto.is_dir():
        raise FileNotFoundError(
            f"Conjunto obrigatorio nao encontrado: {diretorio_conjunto.resolve()}"
        )
    return {
        item.name for item in diretorio_conjunto.iterdir() if item.is_dir()
    }


def validar_dataset(raiz_dataset: Path) -> tuple[list[str], dict, dict]:
    """Valida os tres conjuntos e devolve classes, contagens e ignorados."""
    diretorios_conjuntos = {
        "train": raiz_dataset / "train",
        "val": raiz_dataset / "val",
        "test": raiz_dataset / "test",
    }
    classes_por_conjunto = {
        conjunto: listar_diretorios_classes(diretorio)
        for conjunto, diretorio in diretorios_conjuntos.items()
    }

    classes_treino = classes_por_conjunto["train"]
    if not classes_treino:
        raise RuntimeError("Nenhuma classe encontrada no conjunto train.")

    inconsistencias = {
        conjunto: sorted(classes ^ classes_treino)
        for conjunto, classes in classes_por_conjunto.items()
        if classes != classes_treino
    }
    if inconsistencias:
        raise RuntimeError(
            "As classes de train, val e test nao sao identicas: "
            + json.dumps(inconsistencias, ensure_ascii=False)
        )

    nomes_classes = sorted(classes_treino)
    contagens_por_conjunto = {}
    ignorados_por_conjunto = {}
    for conjunto, diretorio in diretorios_conjuntos.items():
        _, _, contagens, ignorados = base.examinar_caminhos_imagens(
            diretorio, nomes_classes
        )
        classes_vazias = [
            nome for nome, quantidade in contagens.items() if quantidade == 0
        ]
        if classes_vazias:
            raise RuntimeError(
                f"Classes sem imagens compativeis em {conjunto}: "
                + ", ".join(classes_vazias)
            )
        contagens_por_conjunto[conjunto] = contagens
        ignorados_por_conjunto[conjunto] = ignorados

    return nomes_classes, contagens_por_conjunto, ignorados_por_conjunto


def imprimir_resumo_dataset(
    identificador_dataset: str,
    raiz_dataset: Path,
    nomes_classes: list[str],
    contagens_por_conjunto: dict,
) -> None:
    """Exibe caminho, classes e contagens do dataset validado."""
    print(f"Dataset selecionado: {identificador_dataset}")
    print(f"Caminho: {raiz_dataset.resolve()}")
    print(f"Classes detectadas: {len(nomes_classes)}")
    for indice, nome_classe in enumerate(nomes_classes):
        print(
            f"  {indice}: {nome_classe} | "
            f"train={contagens_por_conjunto['train'][nome_classe]}, "
            f"val={contagens_por_conjunto['val'][nome_classe]}, "
            f"test={contagens_por_conjunto['test'][nome_classe]}"
        )


def detectar_dispositivos(exigir_gpu: bool) -> list:
    """Detecta GPUs e aplica a exigencia solicitada pelo usuario."""
    gpus = base.tf.config.list_physical_devices("GPU")
    if not gpus and exigir_gpu:
        raise RuntimeError(
            "Nenhuma GPU foi detectada. Use WSL2 com TensorFlow CUDA ou retire "
            "a opcao --require-gpu."
        )
    if not gpus:
        print("AVISO: nenhuma GPU detectada; o treinamento continuara em CPU.")
    return gpus


def construir_modelo_generico(numero_classes: int, identificador_dataset: str):
    """Constroi o modelo e adiciona o identificador do dataset ao seu nome."""
    modelo_original, modelo_base = base.construir_modelo(numero_classes)
    modelo = keras.Model(
        inputs=modelo_original.inputs,
        outputs=modelo_original.outputs,
        name=(
            "MobileNetV2_Tomato_Modelo_B_"
            f"{criar_identificador_seguro(identificador_dataset)}"
        ),
    )
    return modelo, modelo_base


def salvar_configuracao_treinamento(
    diretorio_saida: Path,
    argumentos: argparse.Namespace,
    identificador_dataset: str,
    raiz_dataset: Path,
    nomes_classes: list[str],
    gpus: list,
) -> None:
    """Salva ambiente e hiperparametros da execucao."""
    configuracao = {
        "dataset_id": identificador_dataset,
        "dataset_root": str(raiz_dataset.resolve()),
        "img_size": base.TAMANHO_IMAGEM,
        "batch_size": argumentos.batch,
        "epochs_head": argumentos.epochs_head,
        "epochs_fine1": argumentos.epochs_fine1,
        "epochs_fine2": argumentos.epochs_fine2,
        "learning_rate_head": base.TAXA_APRENDIZADO_CABECALHO,
        "learning_rate_fine1": base.TAXA_APRENDIZADO_AJUSTE_1,
        "learning_rate_fine2": base.TAXA_APRENDIZADO_AJUSTE_2,
        "fine_tune_at_1": base.INICIO_AJUSTE_FINO_1,
        "fine_tune_at_2": base.INICIO_AJUSTE_FINO_2,
        "use_class_weight": base.USAR_PESOS_DE_CLASSES,
        "cache_dataset": base.ARMAZENAR_DATASET_EM_CACHE,
        "seed": base.SEMENTE,
        "classes": nomes_classes,
        "tensorflow_version": base.tf.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
        "gpus": [str(gpu) for gpu in gpus],
    }
    caminho = diretorio_saida / "metricas" / "training_config.json"
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(configuracao, arquivo, ensure_ascii=False, indent=2)


def principal() -> None:
    """Valida o dataset e executa as tres fases de treinamento."""
    argumentos = analisar_argumentos()
    identificador_dataset, raiz_dataset = resolver_dataset(argumentos)

    diretorio_saida = None
    if not argumentos.check_only:
        nome_execucao = datetime.now().strftime(
            f"{PREFIXO_EXECUCAO}_{criar_identificador_seguro(identificador_dataset)}_"
            "%Y%m%d_%H%M%S"
        )
        diretorio_saida = argumentos.output_dir or (RAIZ_SAIDAS / nome_execucao)
        diretorio_saida = diretorio_saida.expanduser().resolve()
        base.criar_diretorios(diretorio_saida)
        iniciar_captura_console(diretorio_saida)

    nomes_classes, contagens_por_conjunto, ignorados_por_conjunto = validar_dataset(
        raiz_dataset
    )
    imprimir_resumo_dataset(
        identificador_dataset,
        raiz_dataset,
        nomes_classes,
        contagens_por_conjunto,
    )
    if argumentos.check_only:
        print("Dataset validado com sucesso. Treinamento nao iniciado.")
        return

    base.configurar_ambiente()
    gpus = detectar_dispositivos(argumentos.require_gpu)
    diretorio_externo = raiz_dataset / "external_test"

    caminhos_treino, rotulos_treino, _, _ = base.examinar_caminhos_imagens(
        raiz_dataset / "train", nomes_classes
    )
    caminhos_validacao, rotulos_validacao, _, _ = base.examinar_caminhos_imagens(
        raiz_dataset / "val", nomes_classes
    )
    caminhos_teste, rotulos_teste, _, _ = base.examinar_caminhos_imagens(
        raiz_dataset / "test", nomes_classes
    )

    caminhos_externos = np.array([], dtype=str)
    rotulos_externos = np.array([], dtype=np.int32)
    contagens_externas = {nome: 0 for nome in nomes_classes}
    ignorados_externos = []
    if base.teste_externo_esta_rotulado(diretorio_externo, nomes_classes):
        (
            caminhos_externos,
            rotulos_externos,
            contagens_externas,
            ignorados_externos,
        ) = base.examinar_caminhos_imagens(diretorio_externo, nomes_classes)

    base.salvar_relatorio_dataset(
        diretorio_saida,
        nomes_classes,
        {**contagens_por_conjunto, "external_test": contagens_externas},
        {**ignorados_por_conjunto, "external_test": ignorados_externos},
    )
    salvar_configuracao_treinamento(
        diretorio_saida,
        argumentos,
        identificador_dataset,
        raiz_dataset,
        nomes_classes,
        gpus,
    )

    diretorio_modelos = diretorio_saida / "modelos"
    caminho_classes = diretorio_modelos / "classes.txt"
    with caminho_classes.open("w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(nomes_classes) + "\n")
    with (diretorio_modelos / "classes.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            {indice: nome for indice, nome in enumerate(nomes_classes)},
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    dataset_treino = base.criar_dataset(
        caminhos_treino,
        rotulos_treino,
        argumentos.batch,
        treinamento=True,
        cache=base.ARMAZENAR_DATASET_EM_CACHE,
    )
    dataset_validacao = base.criar_dataset(
        caminhos_validacao,
        rotulos_validacao,
        argumentos.batch,
        cache=base.ARMAZENAR_DATASET_EM_CACHE,
    )
    dataset_teste = base.criar_dataset(
        caminhos_teste, rotulos_teste, argumentos.batch
    )
    dataset_externo = (
        base.criar_dataset(
            caminhos_externos, rotulos_externos, argumentos.batch
        )
        if len(caminhos_externos)
        else None
    )

    pesos_classes = None
    if base.USAR_PESOS_DE_CLASSES:
        pesos = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(nomes_classes)),
            y=rotulos_treino,
        )
        pesos_classes = {
            indice: float(peso) for indice, peso in enumerate(pesos)
        }
        with (diretorio_saida / "metricas" / "class_weights.json").open(
            "w", encoding="utf-8"
        ) as arquivo:
            json.dump(pesos_classes, arquivo, indent=2)

    modelo, modelo_base = construir_modelo_generico(
        len(nomes_classes), identificador_dataset
    )
    historicos = []
    fases = (
        (
            "head",
            None,
            base.TAXA_APRENDIZADO_CABECALHO,
            argumentos.epochs_head,
        ),
        (
            "fine_1",
            base.INICIO_AJUSTE_FINO_1,
            base.TAXA_APRENDIZADO_AJUSTE_1,
            argumentos.epochs_fine1,
        ),
        (
            "fine_2",
            base.INICIO_AJUSTE_FINO_2,
            base.TAXA_APRENDIZADO_AJUSTE_2,
            argumentos.epochs_fine2,
        ),
    )
    for nome_fase, inicio_ajuste, taxa_aprendizado, epocas in fases:
        if inicio_ajuste is not None:
            base.configurar_ajuste_fino(modelo_base, inicio_ajuste)
        base.compilar_modelo(modelo, taxa_aprendizado)
        historico = modelo.fit(
            dataset_treino,
            validation_data=dataset_validacao,
            epochs=epocas,
            callbacks=base.criar_callbacks(diretorio_saida, nome_fase),
            class_weight=pesos_classes,
            verbose=1,
        )
        historicos.append((nome_fase, historico))

    base.plotar_historico_treinamento(diretorio_saida, historicos)
    nome_modelo = (
        "tomato_modelo_b_"
        f"{criar_identificador_seguro(identificador_dataset)}_final"
    )
    modelo.save(diretorio_modelos / f"{nome_modelo}.keras")
    if argumentos.export_h5:
        caminho_h5 = diretorio_modelos / f"{nome_modelo}.h5"
        try:
            modelo.save(caminho_h5)
        except Exception as erro:
            caminho_h5.unlink(missing_ok=True)
            print(f"Aviso: nao foi possivel salvar o formato H5: {erro}")

    metricas = [
        base.avaliar_e_salvar(
            diretorio_saida, modelo, dataset_validacao, nomes_classes, "val"
        ),
        base.avaliar_e_salvar(
            diretorio_saida, modelo, dataset_teste, nomes_classes, "test"
        ),
    ]
    if dataset_externo is not None:
        metricas.append(
            base.avaliar_e_salvar(
                diretorio_saida,
                modelo,
                dataset_externo,
                nomes_classes,
                "external_test_labeled",
            )
        )
    with (diretorio_saida / "metricas" / "metrics_summary.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    shutil.copy(caminho_classes, diretorio_saida / "classes.txt")
    print(f"Treinamento concluido. Arquivos: {diretorio_saida.resolve()}")


if __name__ == "__main__":
    inicio = time.time()
    try:
        principal()
    finally:
        tempo_decorrido = time.time() - inicio
        print(f"Tempo total: {tempo_decorrido / 60:.2f} minutos")
