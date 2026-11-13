# Treinamento da IA com MobileNetV2

Código de treinamento do modelo de classificação de doenças em folhas de
tomateiro desenvolvido para o TCC **Sistema Inteligente de Monitoramento
Agrícola e Detecção Precoce de Doenças em Folhas de Tomate**.

O repositório contém dois treinadores executáveis. O treinador principal
valida rigorosamente os três conjuntos, registra o console e permite exigir
GPU. O treinador de dataset expõe diretamente o pipeline original. Ambos
preservam a arquitetura MobileNetV2, o aumento de dados, os pesos de classes,
as três fases de treinamento, o ajuste fino, os callbacks e as métricas.

As imagens do dataset não são distribuídas pelo Git. Somente a estrutura de
pastas e os arquivos `.gitkeep` são versionados.

## Requisitos

- Python 3.12;
- ambiente virtual recomendado;
- espaço em disco suficiente para dataset, checkpoints, modelos e gráficos.

Crie e ative um ambiente virtual e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para um ambiente Linux ou WSL2 com NVIDIA e CUDA compatível:

```bash
python -m pip install -r requirements-gpu.txt
```

No Windows nativo, versões modernas do TensorFlow normalmente executam em
CPU. Para treinamento com GPU NVIDIA, use Linux ou WSL2.

## Estrutura do dataset

Coloque as imagens em `dataset/`, mantendo as mesmas dez classes em `train`,
`val` e `test`:

```text
dataset/
  train/
    Tomato___Bacterial_spot/
    Tomato___Early_blight/
    Tomato___healthy/
    Tomato___Late_blight/
    Tomato___Leaf_Mold/
    Tomato___Septoria_leaf_spot/
    Tomato___Spider_mites Two-spotted_spider_mite/
    Tomato___Target_Spot/
    Tomato___Tomato_mosaic_virus/
    Tomato___Tomato_Yellow_Leaf_Curl_Virus/
  val/
    <as mesmas dez classes>/
  test/
    <as mesmas dez classes>/
```

## Treinador principal

Valide o dataset sem iniciar o treinamento:

```powershell
python .\codigo\treinamento\treinar_modelo_tomate.py --check-only
```

Inicie um treinamento com a configuração padrão:

```powershell
python .\codigo\treinamento\treinar_modelo_tomate.py
```

Exemplo com opções:

```powershell
python .\codigo\treinamento\treinar_modelo_tomate.py `
  --dataset D:\dados\tomates `
  --batch 16 `
  --epochs-head 20 `
  --epochs-fine1 25 `
  --epochs-fine2 15 `
  --require-gpu `
  --export-h5
```
## Treinador de dataset

O segundo comando executa diretamente o pipeline de treinamento:

```powershell
python .\codigo\treinamento\treinar_modelo_tomate_dataset.py
```

Ele aceita `--dataset`, `--batch`, `--epochs_head`, `--epochs_fine1` e
`--epochs_fine2`. Use `--help` para consultar os detalhes.

## Arquivos gerados

Cada execução gera uma pasta datada contendo:

- `modelos/`: checkpoints, modelo final `.keras`, classes em TXT e JSON;
- `metricas/`: configuração, pesos, histórico, métricas, relatórios e matrizes
  de confusão em CSV;
- `graficos/`: curvas de treinamento e matrizes de confusão em PNG;
- `logs/`: CSV de cada fase e, no treinador principal, `console.log`.

O formato oficial do modelo é `.keras`. Resultados podem variar ligeiramente
entre CPU, GPU, CUDA e implementações de kernels, mesmo com a seed preservada.
