"""Treina o Modelo B MobileNetV2 em um dataset de folhas de tomateiro.

O dataset deve conter ``train``, ``val`` e ``test`` organizados em
subdiretorios por classe. ``external_test`` e opcional e so e avaliado quando
tambem esta organizado pelas mesmas classes.
"""

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


SEMENTE = 42

RAIZ_DATASET_PADRAO = Path("dataset")
RAIZ_SAIDAS = Path("outputs_treinamento")
PREFIXO_EXECUCAO = "modelo_b_mobilenetv2"

TAMANHO_IMAGEM = 224
TAMANHO_LOTE_PADRAO = 32

EPOCAS_CABECALHO = 20
EPOCAS_AJUSTE_FINO_1 = 25
EPOCAS_AJUSTE_FINO_2 = 15

TAXA_APRENDIZADO_CABECALHO = 1e-3
TAXA_APRENDIZADO_AJUSTE_1 = 1e-5
TAXA_APRENDIZADO_AJUSTE_2 = 3e-6

INICIO_AJUSTE_FINO_1 = 100
INICIO_AJUSTE_FINO_2 = 70

USAR_PRECISAO_MISTA = True
USAR_XLA = False
ARMAZENAR_DATASET_EM_CACHE = False
USAR_PESOS_DE_CLASSES = True

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configurar_ambiente():
    """Configura sementes, GPU, precisao mista e XLA."""
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    tf.random.set_seed(SEMENTE)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs detectadas: {gpus}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as erro:
                print(f"Aviso: nao foi possivel ativar crescimento de memoria: {erro}")
    else:
        print("Nenhuma GPU detectada pelo TensorFlow. O treino sera executado em CPU.")

    if USAR_PRECISAO_MISTA and gpus:
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("Precisao mista ativada: mixed_float16")
        except Exception as erro:
            print(f"Aviso: nao foi possivel ativar precisao mista: {erro}")

    if USAR_XLA:
        try:
            tf.config.optimizer.set_jit(True)
            print("XLA/JIT ativado.")
        except Exception as erro:
            print(f"Aviso: nao foi possivel ativar XLA: {erro}")


def criar_diretorios(diretorio_saida: Path):
    """Cria a arvore de diretorios de uma execucao."""
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    (diretorio_saida / "modelos").mkdir(exist_ok=True)
    (diretorio_saida / "metricas").mkdir(exist_ok=True)
    (diretorio_saida / "graficos").mkdir(exist_ok=True)
    (diretorio_saida / "logs").mkdir(exist_ok=True)


def descobrir_nomes_classes(diretorio_treino: Path):
    """Devolve os nomes das classes encontrados no conjunto de treino."""
    if not diretorio_treino.exists():
        raise FileNotFoundError(
            f"Diretorio de treino nao encontrado: {diretorio_treino.resolve()}"
        )

    classes = sorted(
        item.name for item in diretorio_treino.iterdir() if item.is_dir()
    )
    if not classes:
        raise RuntimeError(
            f"Nenhuma classe encontrada em: {diretorio_treino.resolve()}"
        )
    return classes


def examinar_caminhos_imagens(diretorio_raiz: Path, nomes_classes):
    """Coleta imagens, rotulos, contagens e diretorios ignorados."""
    caminhos = []
    rotulos = []
    contagens = {classe: 0 for classe in nomes_classes}
    diretorios_ignorados = []

    if not diretorio_raiz.exists():
        return (
            np.array([], dtype=str),
            np.array([], dtype=np.int32),
            contagens,
            diretorios_ignorados,
        )

    indice_por_classe = {
        classe: indice for indice, classe in enumerate(nomes_classes)
    }
    for item in sorted(diretorio_raiz.iterdir()):
        if item.is_dir() and item.name not in indice_por_classe:
            diretorios_ignorados.append(item.name)

    for nome_classe in nomes_classes:
        diretorio_classe = diretorio_raiz / nome_classe
        if not diretorio_classe.exists():
            continue
        for arquivo in diretorio_classe.rglob("*"):
            if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS:
                caminhos.append(str(arquivo))
                rotulos.append(indice_por_classe[nome_classe])
                contagens[nome_classe] += 1

    return (
        np.array(caminhos),
        np.array(rotulos, dtype=np.int32),
        contagens,
        diretorios_ignorados,
    )


def teste_externo_esta_rotulado(diretorio_externo: Path, nomes_classes):
    """Informa se o teste externo possui ao menos uma pasta de classe."""
    if not diretorio_externo.exists():
        return False
    return any((diretorio_externo / classe).exists() for classe in nomes_classes)


def carregar_e_pre_processar_imagem(caminho, rotulo):
    """Le e redimensiona uma imagem sem alterar sua escala de pixels."""
    bytes_imagem = tf.io.read_file(caminho)
    imagem = tf.io.decode_image(bytes_imagem, channels=3, expand_animations=False)
    imagem.set_shape([None, None, 3])
    imagem = tf.image.resize(
        imagem, [TAMANHO_IMAGEM, TAMANHO_IMAGEM], method="bilinear"
    )
    imagem = tf.cast(imagem, tf.float32)
    return imagem, rotulo


def criar_dataset(caminhos, rotulos, tamanho_lote, treinamento=False, cache=False):
    """Monta o pipeline ``tf.data`` usado pelo modelo."""
    dataset = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    if treinamento:
        tamanho_buffer = min(len(caminhos), 10000)
        dataset = dataset.shuffle(
            buffer_size=tamanho_buffer,
            seed=SEMENTE,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        carregar_e_pre_processar_imagem,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if cache:
        dataset = dataset.cache()
    return dataset.batch(tamanho_lote).prefetch(tf.data.AUTOTUNE)


def salvar_relatorio_dataset(
    diretorio_saida,
    nomes_classes,
    contagens_por_conjunto,
    ignorados_por_conjunto,
):
    """Salva as contagens e os diretorios ignorados durante a leitura."""
    caminho_relatorio = diretorio_saida / "metricas" / "dataset_report.csv"
    with caminho_relatorio.open("w", encoding="utf-8") as arquivo:
        arquivo.write("classe,train,val,test,external_test_labeled\n")
        for classe in nomes_classes:
            arquivo.write(
                f"{classe},"
                f"{contagens_por_conjunto['train'].get(classe, 0)},"
                f"{contagens_por_conjunto['val'].get(classe, 0)},"
                f"{contagens_por_conjunto['test'].get(classe, 0)},"
                f"{contagens_por_conjunto['external_test'].get(classe, 0)}\n"
            )

    caminho_ignorados = diretorio_saida / "metricas" / "ignored_dirs.json"
    with caminho_ignorados.open("w", encoding="utf-8") as arquivo:
        json.dump(ignorados_por_conjunto, arquivo, ensure_ascii=False, indent=2)
    print(f"Relatorio do dataset salvo em: {caminho_relatorio}")


def criar_aumento_dados():
    """Cria o aumento moderado de dados aplicado a imagens de folhas."""
    camadas_aumento = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(
            height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)
        ),
        layers.RandomTranslation(height_factor=0.06, width_factor=0.06),
        layers.RandomContrast(0.15),
    ]
    if hasattr(layers, "RandomBrightness"):
        camadas_aumento.append(layers.RandomBrightness(factor=0.10))
    return keras.Sequential(camadas_aumento, name="data_augmentation")


def construir_modelo(numero_classes):
    """Constroi a MobileNetV2 e seu cabecalho classificador."""
    entradas = keras.Input(
        shape=(TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3), name="input_image"
    )
    saida = criar_aumento_dados()(entradas)
    saida = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_rescale_minus1_1",
    )(saida)

    modelo_base = MobileNetV2(
        input_shape=(TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3),
        include_top=False,
        weights="imagenet",
        alpha=1.0,
    )
    modelo_base.trainable = False

    saida = modelo_base(saida, training=False)
    saida = layers.GlobalAveragePooling2D(name="global_average_pooling")(saida)
    saida = layers.BatchNormalization(name="bn_1")(saida)
    saida = layers.Dropout(0.45, name="dropout_1")(saida)
    saida = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_256",
    )(saida)
    saida = layers.BatchNormalization(name="bn_2")(saida)
    saida = layers.Dropout(0.35, name="dropout_2")(saida)
    saidas = layers.Dense(
        numero_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(saida)

    modelo = keras.Model(
        inputs=entradas,
        outputs=saidas,
        name="MobileNetV2_Tomato_Modelo_B_Dataset",
    )
    return modelo, modelo_base


def compilar_modelo(modelo, taxa_aprendizado):
    """Compila o modelo com as metricas usadas no treinamento."""
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=2, name="top2_accuracy"
            ),
        ],
    )


def criar_callbacks(diretorio_saida: Path, nome_fase: str):
    """Cria checkpoint, parada antecipada, agenda e log CSV da fase."""
    caminho_checkpoint = diretorio_saida / "modelos" / f"best_{nome_fase}.keras"
    caminho_log_csv = diretorio_saida / "logs" / f"training_{nome_fase}.csv"
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(caminho_checkpoint),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.35,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(caminho_log_csv), append=False),
    ]


def configurar_ajuste_fino(modelo_base, inicio_ajuste):
    """Descongela a base a partir da camada indicada, exceto BatchNorm."""
    modelo_base.trainable = True
    for camada in modelo_base.layers[:inicio_ajuste]:
        camada.trainable = False
    for camada in modelo_base.layers:
        if isinstance(camada, layers.BatchNormalization):
            camada.trainable = False

    total_treinaveis = sum(1 for camada in modelo_base.layers if camada.trainable)
    print(
        "Ajuste fino ativado. Camadas treinaveis no modelo base: "
        f"{total_treinaveis}/{len(modelo_base.layers)}"
    )


def predizer_dataset(modelo, dataset):
    """Executa a predicao e devolve rotulos reais, previstos e probabilidades."""
    rotulos_reais = []
    probabilidades = []
    for lote_imagens, lote_rotulos in dataset:
        probabilidades_lote = modelo.predict(lote_imagens, verbose=0)
        probabilidades.append(probabilidades_lote)
        rotulos_reais.append(lote_rotulos.numpy())

    rotulos_reais = np.concatenate(rotulos_reais)
    probabilidades = np.concatenate(probabilidades)
    rotulos_previstos = np.argmax(probabilidades, axis=1)
    return rotulos_reais, rotulos_previstos, probabilidades


def avaliar_e_salvar(
    diretorio_saida,
    modelo,
    dataset,
    nomes_classes,
    nome_conjunto,
):
    """Calcula e salva metricas, relatorio e matriz de confusao."""
    print(f"\nAvaliando conjunto: {nome_conjunto}")
    rotulos_reais, rotulos_previstos, _ = predizer_dataset(modelo, dataset)
    todos_rotulos = list(range(len(nomes_classes)))
    rotulos_presentes = sorted(np.unique(rotulos_reais).tolist())
    nomes_presentes = [nomes_classes[indice] for indice in rotulos_presentes]

    metricas = {
        "split": nome_conjunto,
        "accuracy": float(accuracy_score(rotulos_reais, rotulos_previstos)),
        "precision_macro": float(
            precision_score(
                rotulos_reais, rotulos_previstos, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                rotulos_reais, rotulos_previstos, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                rotulos_reais, rotulos_previstos, average="macro", zero_division=0
            )
        ),
        "precision_weighted": float(
            precision_score(
                rotulos_reais,
                rotulos_previstos,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                rotulos_reais,
                rotulos_previstos,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                rotulos_reais,
                rotulos_previstos,
                average="weighted",
                zero_division=0,
            )
        ),
        "samples": int(len(rotulos_reais)),
    }

    caminho_metricas = diretorio_saida / "metricas" / f"metrics_{nome_conjunto}.json"
    with caminho_metricas.open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    relatorio = classification_report(
        rotulos_reais,
        rotulos_previstos,
        labels=rotulos_presentes,
        target_names=nomes_presentes,
        zero_division=0,
    )
    caminho_relatorio = (
        diretorio_saida
        / "metricas"
        / f"classification_report_{nome_conjunto}.txt"
    )
    with caminho_relatorio.open("w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)

    matriz = confusion_matrix(
        rotulos_reais, rotulos_previstos, labels=todos_rotulos
    )
    np.savetxt(
        diretorio_saida / "metricas" / f"confusion_matrix_{nome_conjunto}.csv",
        matriz,
        delimiter=",",
        fmt="%d",
    )
    plotar_matriz_confusao(
        diretorio_saida, matriz, nomes_classes, nome_conjunto
    )

    print(json.dumps(metricas, ensure_ascii=False, indent=2))
    print(relatorio)
    return metricas


def plotar_matriz_confusao(
    diretorio_saida,
    matriz,
    nomes_classes,
    nome_conjunto,
):
    """Salva a matriz de confusao como imagem PNG."""
    tamanho_figura = max(9, len(nomes_classes) * 0.85)
    plt.figure(figsize=(tamanho_figura, tamanho_figura))
    plt.imshow(matriz, interpolation="nearest")
    plt.title(f"Matriz de confusao - {nome_conjunto}")
    plt.colorbar()

    marcacoes = np.arange(len(nomes_classes))
    plt.xticks(marcacoes, nomes_classes, rotation=90)
    plt.yticks(marcacoes, nomes_classes)
    plt.ylabel("Classe real")
    plt.xlabel("Classe prevista")
    plt.tight_layout()
    plt.savefig(
        diretorio_saida / "graficos" / f"confusion_matrix_{nome_conjunto}.png",
        dpi=180,
    )
    plt.close()


def plotar_historico_treinamento(diretorio_saida, historicos):
    """Combina as fases e salva historico e curvas de treinamento."""
    historico_combinado = {}
    for _, historico in historicos:
        for chave, valores in historico.history.items():
            historico_combinado.setdefault(chave, []).extend(valores)

    caminho_historico = diretorio_saida / "metricas" / "history.json"
    with caminho_historico.open("w", encoding="utf-8") as arquivo:
        json.dump(
            {
                chave: [float(valor) for valor in valores]
                for chave, valores in historico_combinado.items()
            },
            arquivo,
            indent=2,
        )

    for metrica in ["accuracy", "loss", "top2_accuracy"]:
        metrica_validacao = f"val_{metrica}"
        if (
            metrica not in historico_combinado
            or metrica_validacao not in historico_combinado
        ):
            continue
        plt.figure(figsize=(10, 6))
        plt.plot(historico_combinado[metrica], label=f"train_{metrica}")
        plt.plot(
            historico_combinado[metrica_validacao], label=metrica_validacao
        )
        plt.title(f"Evolucao de {metrica}")
        plt.xlabel("Epoca")
        plt.ylabel(metrica)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            diretorio_saida / "graficos" / f"history_{metrica}.png", dpi=180
        )
        plt.close()


def analisar_argumentos():
    """Le os argumentos de linha de comando do treinador independente."""
    analisador = argparse.ArgumentParser(
        description="Treina o Modelo B MobileNetV2 em um dataset de tomateiros."
    )
    analisador.add_argument(
        "--dataset",
        type=str,
        default=str(RAIZ_DATASET_PADRAO),
        help="Caminho do dataset (padrao: %(default)s).",
    )
    analisador.add_argument(
        "--batch", type=int, default=TAMANHO_LOTE_PADRAO, help="Tamanho do lote."
    )
    analisador.add_argument(
        "--epochs_head",
        type=int,
        default=EPOCAS_CABECALHO,
        help="Epocas da fase do cabecalho.",
    )
    analisador.add_argument(
        "--epochs_fine1",
        type=int,
        default=EPOCAS_AJUSTE_FINO_1,
        help="Epocas do primeiro ajuste fino.",
    )
    analisador.add_argument(
        "--epochs_fine2",
        type=int,
        default=EPOCAS_AJUSTE_FINO_2,
        help="Epocas do segundo ajuste fino.",
    )
    return analisador.parse_args()


def principal():
    """Executa leitura, treinamento em tres fases e avaliacao."""
    argumentos = analisar_argumentos()
    configurar_ambiente()

    raiz_dataset = Path(argumentos.dataset)
    diretorio_treino = raiz_dataset / "train"
    diretorio_validacao = raiz_dataset / "val"
    diretorio_teste = raiz_dataset / "test"
    diretorio_externo = raiz_dataset / "external_test"

    nome_execucao = datetime.now().strftime(f"{PREFIXO_EXECUCAO}_%Y%m%d_%H%M%S")
    diretorio_saida = RAIZ_SAIDAS / nome_execucao
    criar_diretorios(diretorio_saida)

    print(f"Dataset: {raiz_dataset.resolve()}")
    print(f"Saida do treinamento: {diretorio_saida.resolve()}")

    nomes_classes = descobrir_nomes_classes(diretorio_treino)
    numero_classes = len(nomes_classes)
    print(f"\nClasses detectadas ({numero_classes}):")
    for indice, classe in enumerate(nomes_classes):
        print(f"  {indice}: {classe}")

    caminho_classes_txt = diretorio_saida / "modelos" / "classes.txt"
    with caminho_classes_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(nomes_classes) + "\n")
    with (diretorio_saida / "modelos" / "classes.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            {indice: classe for indice, classe in enumerate(nomes_classes)},
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    configuracao = {
        "dataset_root": str(raiz_dataset),
        "train_dir": str(diretorio_treino),
        "val_dir": str(diretorio_validacao),
        "test_dir": str(diretorio_teste),
        "external_dir": str(diretorio_externo),
        "img_size": TAMANHO_IMAGEM,
        "batch_size": argumentos.batch,
        "epochs_head": argumentos.epochs_head,
        "epochs_fine1": argumentos.epochs_fine1,
        "epochs_fine2": argumentos.epochs_fine2,
        "learning_rate_head": TAXA_APRENDIZADO_CABECALHO,
        "learning_rate_fine1": TAXA_APRENDIZADO_AJUSTE_1,
        "learning_rate_fine2": TAXA_APRENDIZADO_AJUSTE_2,
        "fine_tune_at_1": INICIO_AJUSTE_FINO_1,
        "fine_tune_at_2": INICIO_AJUSTE_FINO_2,
        "use_class_weight": USAR_PESOS_DE_CLASSES,
        "cache_dataset": ARMAZENAR_DATASET_EM_CACHE,
        "classes": nomes_classes,
    }
    with (diretorio_saida / "metricas" / "training_config.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(configuracao, arquivo, ensure_ascii=False, indent=2)

    caminhos_treino, rotulos_treino, contagens_treino, ignorados_treino = (
        examinar_caminhos_imagens(diretorio_treino, nomes_classes)
    )
    caminhos_validacao, rotulos_validacao, contagens_validacao, ignorados_validacao = (
        examinar_caminhos_imagens(diretorio_validacao, nomes_classes)
    )
    caminhos_teste, rotulos_teste, contagens_teste, ignorados_teste = (
        examinar_caminhos_imagens(diretorio_teste, nomes_classes)
    )

    caminhos_externos = np.array([], dtype=str)
    rotulos_externos = np.array([], dtype=np.int32)
    contagens_externas = {classe: 0 for classe in nomes_classes}
    ignorados_externos = []
    if teste_externo_esta_rotulado(diretorio_externo, nomes_classes):
        (
            caminhos_externos,
            rotulos_externos,
            contagens_externas,
            ignorados_externos,
        ) = examinar_caminhos_imagens(diretorio_externo, nomes_classes)
    else:
        print("\nTeste externo nao esta rotulado por classes compativeis.")
        print("A avaliacao externa com metricas sera ignorada.")

    contagens_por_conjunto = {
        "train": contagens_treino,
        "val": contagens_validacao,
        "test": contagens_teste,
        "external_test": contagens_externas,
    }
    ignorados_por_conjunto = {
        "train": ignorados_treino,
        "val": ignorados_validacao,
        "test": ignorados_teste,
        "external_test": ignorados_externos,
    }
    salvar_relatorio_dataset(
        diretorio_saida,
        nomes_classes,
        contagens_por_conjunto,
        ignorados_por_conjunto,
    )

    print("\nTotais:")
    print(f"  train: {len(caminhos_treino)}")
    print(f"  val:   {len(caminhos_validacao)}")
    print(f"  test:  {len(caminhos_teste)}")
    print(f"  external_test rotulado: {len(caminhos_externos)}")
    if len(caminhos_treino) == 0 or len(caminhos_validacao) == 0:
        raise RuntimeError("Train e val precisam conter imagens.")

    dataset_treino = criar_dataset(
        caminhos_treino,
        rotulos_treino,
        argumentos.batch,
        treinamento=True,
        cache=ARMAZENAR_DATASET_EM_CACHE,
    )
    dataset_validacao = criar_dataset(
        caminhos_validacao,
        rotulos_validacao,
        argumentos.batch,
        cache=ARMAZENAR_DATASET_EM_CACHE,
    )
    dataset_teste = (
        criar_dataset(caminhos_teste, rotulos_teste, argumentos.batch)
        if len(caminhos_teste)
        else None
    )
    dataset_externo = (
        criar_dataset(caminhos_externos, rotulos_externos, argumentos.batch)
        if len(caminhos_externos)
        else None
    )

    pesos_classes = None
    if USAR_PESOS_DE_CLASSES:
        pesos = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(numero_classes),
            y=rotulos_treino,
        )
        pesos_classes = {
            indice: float(peso) for indice, peso in enumerate(pesos)
        }
        print("\nPesos das classes:")
        print(json.dumps(pesos_classes, indent=2))
        with (diretorio_saida / "metricas" / "class_weights.json").open(
            "w", encoding="utf-8"
        ) as arquivo:
            json.dump(pesos_classes, arquivo, indent=2)

    modelo, modelo_base = construir_modelo(numero_classes)
    modelo.summary()
    historicos = []
    fases = (
        (
            "head",
            "cabecalho classificador",
            None,
            TAXA_APRENDIZADO_CABECALHO,
            argumentos.epochs_head,
        ),
        (
            "fine_1",
            "ajuste fino moderado",
            INICIO_AJUSTE_FINO_1,
            TAXA_APRENDIZADO_AJUSTE_1,
            argumentos.epochs_fine1,
        ),
        (
            "fine_2",
            "ajuste fino mais profundo",
            INICIO_AJUSTE_FINO_2,
            TAXA_APRENDIZADO_AJUSTE_2,
            argumentos.epochs_fine2,
        ),
    )
    for nome_fase, descricao, inicio_ajuste, taxa, epocas in fases:
        print(f"\nFASE {len(historicos) + 1}: {descricao}")
        if inicio_ajuste is not None:
            configurar_ajuste_fino(modelo_base, inicio_ajuste)
        compilar_modelo(modelo, taxa)
        historico = modelo.fit(
            dataset_treino,
            validation_data=dataset_validacao,
            epochs=epocas,
            callbacks=criar_callbacks(diretorio_saida, nome_fase),
            class_weight=pesos_classes,
            verbose=1,
        )
        historicos.append((nome_fase, historico))

    plotar_historico_treinamento(diretorio_saida, historicos)
    caminho_modelo = diretorio_saida / "modelos" / "tomato_modelo_b_dataset_final.keras"
    modelo.save(caminho_modelo)
    print(f"\nModelo final salvo em: {caminho_modelo}")

    try:
        caminho_h5 = diretorio_saida / "modelos" / "tomato_modelo_b_dataset_final.h5"
        modelo.save(caminho_h5)
        print(f"Modelo H5 salvo em: {caminho_h5}")
    except Exception as erro:
        print(f"Aviso: nao foi possivel salvar em H5: {erro}")

    todas_metricas = [
        avaliar_e_salvar(
            diretorio_saida,
            modelo,
            dataset_validacao,
            nomes_classes,
            "val",
        )
    ]
    if dataset_teste is not None:
        todas_metricas.append(
            avaliar_e_salvar(
                diretorio_saida, modelo, dataset_teste, nomes_classes, "test"
            )
        )
    else:
        print("Conjunto test nao encontrado ou vazio. Avaliacao ignorada.")
    if dataset_externo is not None:
        todas_metricas.append(
            avaliar_e_salvar(
                diretorio_saida,
                modelo,
                dataset_externo,
                nomes_classes,
                "external_test_labeled",
            )
        )
    else:
        print("Teste externo rotulado nao encontrado. Avaliacao ignorada.")

    with (diretorio_saida / "metricas" / "metrics_summary.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(todas_metricas, arquivo, ensure_ascii=False, indent=2)

    shutil.copy(caminho_classes_txt, diretorio_saida / "classes.txt")
    print("\nTreinamento do Modelo B concluido.")
    print(f"Arquivos gerados em: {diretorio_saida.resolve()}")


if __name__ == "__main__":
    inicio = time.time()
    try:
        principal()
    finally:
        tempo_decorrido = time.time() - inicio
        print(f"Tempo total: {tempo_decorrido / 60:.2f} minutos")
