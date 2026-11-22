import argparse
import contextlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from nucleo import (
    transformacao, dividir_arquivos, listar_imagens, DatasetReconstrucao,
    AutoencoderCNN, calibrar_lambda, treinar, euler_exato,
    SEED, SEED_VALIDACAO, TEMPERATURA_PADRAO, LIMIAR
)

import optuna


# -----------------------------------------------------------
# Espaco de busca

class Amostrador:

    def __init__(self, trial=None, rng=None):
        self.trial = trial
        self.rng = rng

    def log(self, nome, minimo, maximo):
        if self.trial is not None:
            return self.trial.suggest_float(nome, minimo, maximo, log=True)
        return float(np.exp(self.rng.uniform(np.log(minimo), np.log(maximo))))

    def categorica(self, nome, opcoes):
        if self.trial is not None:
            return self.trial.suggest_categorical(nome, opcoes)
        return self.rng.choice(opcoes)


def sugerir(amostrador, com_euler):

    config = {
        "taxa_aprendizado": amostrador.log("taxa_aprendizado", 1e-4, 3e-3),
        "weight_decay": amostrador.log("weight_decay", 1e-8, 1e-3),
        "patience": amostrador.categorica("patience", [5, 8, 12, 16, 20]),
        "batch_size": amostrador.categorica("batch_size", [8, 16, 32]),
    }

    if com_euler:
        # peso do termo de Euler relativo ao MSE inicial (ver calibrar_lambda)
        config["peso_relativo"] = amostrador.log("peso_relativo", 0.01, 1.0)
        # suavizacao da sigmoide: T pequeno aproxima melhor o chi exato
        config["temperatura"] = amostrador.log("temperatura", 0.005, 0.05)
    else:
        config["peso_relativo"] = 0.0
        config["temperatura"] = TEMPERATURA_PADRAO

    return config


# -----------------------------------------------------------
# Dataset

def preparar_dados(pasta_tile):

    pasta_tile = Path(pasta_tile)

    arquivos_treino, arquivos_validacao = dividir_arquivos(pasta_tile / "train" / "good", seed=SEED)

    treino = DatasetReconstrucao(arquivos_treino, transformacao)

    validacao = DatasetReconstrucao(arquivos_validacao, transformacao, danos_fixos=True, semente=SEED_VALIDACAO)

    teste = None
    pasta_teste = pasta_tile / "test" / "good"

    if pasta_teste.exists():
        teste = DatasetReconstrucao(listar_imagens(pasta_teste), transformacao, danos_fixos=True, semente=SEED_VALIDACAO + 10_000)

    return treino, validacao, teste


# -----------------------------------------------------------
# Avalizacao

def reconstruir(modelo, entradas, dispositivo):

    modelo.eval()
    saidas = []

    with torch.no_grad():
        for i in range(0, len(entradas), 16):
            saidas.append(modelo(entradas[i:i + 16].to(dispositivo)).cpu())

    return torch.cat(saidas)


def avaliar(modelo, dataset, dispositivo):
    """MSE, PSNR, SSIM e erro de chi EXATO (viz. 4, limiar 0.5) de um dataset de danos fixos."""

    entradas = torch.stack([dataset[i][0] for i in range(len(dataset))])
    alvos = torch.stack([dataset[i][1] for i in range(len(dataset))])

    rec = reconstruir(modelo, entradas, dispositivo)

    mses, psnrs, ssims, erros_chi = [], [], [], []

    for r, a in zip(rec, alvos):

        r, a = r[0].numpy(), a[0].numpy()

        mse = float(np.mean((r - a) ** 2))

        mses.append(mse)
        psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        ssims.append(structural_similarity(r, a, data_range=1.0))
        erros_chi.append(abs(euler_exato(r, LIMIAR, 1) - euler_exato(a, LIMIAR, 1)))

    return {
        "mse": float(np.mean(mses)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "erro_chi": float(np.mean(erros_chi)),
    }


# -----------------------------------------------------------
# Treino de um configuraÃ§Ã£o

def treinar_config(config, treino, validacao, epocas, semente, dispositivo,
                   caminho_modelo, silencioso=True):

    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)

    gerador = torch.Generator().manual_seed(semente)

    loader_treino = DataLoader(treino, batch_size=config["batch_size"], shuffle=True, num_workers=0, generator=gerador)
    loader_validacao = DataLoader(validacao, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    if config["peso_relativo"] > 0:
        lambda_euler, _, _ = calibrar_lambda(loader_treino, config["peso_relativo"], temperatura=config["temperatura"])
    else:
        lambda_euler = 0.0

    modelo = AutoencoderCNN().to(dispositivo)

    saida = io.StringIO() if silencioso else None

    with contextlib.redirect_stdout(saida) if silencioso else contextlib.nullcontext():
        historico = treinar(
            modelo, loader_treino, loader_validacao,
            lambda_euler=lambda_euler, epocas=epocas,
            taxa_aprendizado=config["taxa_aprendizado"],
            weight_decay=config["weight_decay"],
            patience=config["patience"],
            dispositivo=dispositivo, caminho_modelo=caminho_modelo,
            temperatura=config["temperatura"]
        )

    return modelo, lambda_euler, historico


# -----------------------------------------------------------
# Busca

def buscar(nome, com_euler, treino, validacao, args, dispositivo, pasta_saida):

    print(f"\n{'=' * 60}\nBUSCA: {nome} | objetivo = {args.objetivo} | "
          f"{args.tentativas} tentativas x {args.epocas_busca} epocas"
          f"\n{'=' * 60}")

    temporario = pasta_saida / "tmp" / f"{nome}_busca.pth"
    registro = []

    def avaliar_config(config):

        inicio = time.time()

        try:
            modelo, lam, _ = treinar_config(config, treino, validacao, args.epocas_busca, SEED, dispositivo, temporario)
            metricas = avaliar(modelo, validacao, dispositivo)
            valor = metricas["mse"] if args.objetivo == "mse" else metricas["erro_chi"]

            if not np.isfinite(valor):
                raise FloatingPointError("valor nao finito")

        except Exception as erro:
            print(f"  tentativa falhou: {erro}")
            return float("inf"), {}, 0.0

        registro.append({"config": config, "lambda": lam,
                         "valor": valor, "metricas": metricas})

        print(f"  {json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in config.items()})}"
              f" -> {args.objetivo} = {valor:.5f} | MSE = {metricas['mse']:.5f}"
              f" | erro chi = {metricas['erro_chi']:.1f} | {time.time() - inicio:.0f}s")

        return valor, metricas, lam

    if optuna is not None:

        def objetivo(trial):
            return avaliar_config(sugerir(Amostrador(trial=trial), com_euler))[0]

        estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        estudo.optimize(objetivo, n_trials=args.tentativas)

    else:

        print("  (optuna nao instalado: busca aleatoria com semente fixa)")

        rng = random.Random(SEED)

        for _ in range(args.tentativas):
            avaliar_config(sugerir(Amostrador(rng=rng), com_euler))

    if not registro:
        raise RuntimeError("Nenhuma tentativa terminou com sucesso.")

    melhor = min(registro, key=lambda r: r["valor"])

    print(f"\nMelhor configuracao ({nome}): {melhor['config']}")
    print(f"  {args.objetivo} de busca = {melhor['valor']:.5f}")

    return melhor["config"], registro


# -----------------------------------------------------------
# Treino final

def treino_final(nome, config, treino, validacao, teste, args, dispositivo, pasta_saida):

    print(f"\nTREINO FINAL: {nome} | {args.epocas_final} epocas | "
          f"{args.sementes} semente(s)")

    resultados = {"validacao": [], "teste": []}

    for k in range(args.sementes):

        semente = SEED + k

        caminho = pasta_saida / "modelos" / f"otimizado_{nome}_s{semente}.pth"

        modelo, lam, _ = treinar_config(config, treino, validacao, args.epocas_final, semente, dispositivo, caminho, silencioso=True)

        resultados["validacao"].append(avaliar(modelo, validacao, dispositivo))

        if teste is not None:
            resultados["teste"].append(avaliar(modelo, teste, dispositivo))

        v = resultados["validacao"][-1]

        print(f"  semente {semente}: lambda = {lam:.2e} | val MSE = {v['mse']:.5f} | "
              f"SSIM = {v['ssim']:.4f} | erro chi = {v['erro_chi']:.1f}")

    return resultados


def resumir(resultados):
    """Media e desvio-padrao entre sementes de cada metrica."""

    resumo = {}

    for conjunto, lista in resultados.items():

        if not lista:
            continue

        resumo[conjunto] = {
            metrica: {
                "media": float(np.mean([r[metrica] for r in lista])),
                "desvio": float(np.std([r[metrica] for r in lista])),
            }
            for metrica in lista[0]
        }

    return resumo


def imprimir_tabela(resumos):

    for conjunto in ["validacao", "teste"]:

        if not all(conjunto in r for r in resumos.values()):
            continue

        print(f"\n--- {conjunto.upper()} (media +/- desvio entre sementes) ---")
        print(f"{'modelo':12s}{'MSE':>20s}{'PSNR (dB)':>18s}{'SSIM':>18s}{'erro chi':>18s}")

        for nome, resumo in resumos.items():

            m = resumo[conjunto]

            print(
                f"{nome:12s}"
                f"{m['mse']['media']:12.5f} +/- {m['mse']['desvio']:.5f}"
                f"{m['psnr']['media']:9.2f} +/- {m['psnr']['desvio']:.2f}"
                f"{m['ssim']['media']:9.4f} +/- {m['ssim']['desvio']:.4f}"
                f"{m['erro_chi']['media']:9.1f} +/- {m['erro_chi']['desvio']:.1f}"
            )


# -----------------------------------------------------------
# Execucao

def principal():

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--tile", default="datasets/MVTecAD/tile")
    parser.add_argument("--saida", default="otimizacao")
    parser.add_argument("--objetivo", choices=["mse", "chi"], default="mse",
                        help="criterio minimizado na busca (igual para os dois modelos)")
    parser.add_argument("--tentativas", type=int, default=20)
    parser.add_argument("--epocas-busca", type=int, default=60)
    parser.add_argument("--epocas-final", type=int, default=200)
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--apenas", choices=["com", "sem", "ambos"], default="ambos")

    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pasta_saida = Path(args.saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    print("Dispositivo:", dispositivo)
    print("Busca:", "Optuna (TPE)" if optuna is not None else "aleatoria")

    treino, validacao, teste = preparar_dados(args.tile)

    print(f"Treino: {len(treino)} | Validacao: {len(validacao)} | "
          f"Teste: {len(teste) if teste is not None else 'ausente (test/good nao encontrado)'}")

    modelos = []

    if args.apenas in ("sem", "ambos"):
        modelos.append(("sem_euler", False))

    if args.apenas in ("com", "ambos"):
        modelos.append(("com_euler", True))

    relatorio, resumos = {}, {}

    for nome, com_euler in modelos:

        config, registro = buscar(nome, com_euler, treino, validacao, args, dispositivo, pasta_saida)

        resultados = treino_final(nome, config, treino, validacao, teste, args, dispositivo, pasta_saida)

        resumos[nome] = resumir(resultados)

        relatorio[nome] = {
            "melhor_config": config,
            "busca": registro,
            "final_por_semente": resultados,
            "final_resumo": resumos[nome],
        }

        with open(pasta_saida / "otimizacao_resultados.json", "w") as f:
            json.dump({"args": vars(args), "modelos": relatorio}, f, indent=2)

    imprimir_tabela(resumos)

    print("\nResultados salvos em:", (pasta_saida / "otimizacao_resultados.json").resolve())


if __name__ == "__main__":
    principal()
