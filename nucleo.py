"""
nucleo.py - funcoes compartilhadas entre os notebooks.
"""
# -----------------------------------------------------------
# Importacoes

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.measure import euler_number

# -----------------------------------------------------------
# Constantes compartilhadas

TAMANHO = 256
SEED = 18
SEED_VALIDACAO = 20
TEMPERATURA_PADRAO = 0.02
LIMIAR = 0.5

transformacao = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((TAMANHO, TAMANHO)),
    transforms.ToTensor()
])


# -----------------------------------------------------------
# Calculo da caracteristica de euler


def euler_exato(imagem, limiar=LIMIAR, conectividade=1):
    mascara = np.asarray(imagem) > limiar
    # funÃ§Ã£o que devolve o euler exato ("euler_number" - scikit-image) para uma imagem 2D
    return int(euler_number(mascara, connectivity=conectividade))


def caracteristica_euler_suave_4(imagem, temperatura=TEMPERATURA_PADRAO):
    """
    Caracteristica de Euler diferenciavel (vizinhanca 4).
    Cada pixel ocupado Ã© um vÃ©rtice:
        V = pixels ocupados
        E = pares de pixels vizinhos (horizontal e vertical) - representa quantidade de arestas
        F = blocos 2x2 totalmente ocupados (as faces)
        chi = V - E + F
    """

    # Intensidade dos pixels --> valores de 0 a 1
    p = torch.sigmoid((imagem - 0.5) / temperatura)

    # Soma as probabilidades de ocupaÃ§Ã£o de todos os pixels
    vertices = p.sum(dim=(1, 2, 3))

    # Produto entre dois pixels vizinhos = probabilidade aproximada de estarem ocupados simultaneamente
    arestas_h = (p[:, :, :, :-1] * p[:, :, :, 1:]).sum(dim=(1, 2, 3))
    arestas_v = (p[:, :, :-1, :] * p[:, :, 1:, :]).sum(dim=(1, 2, 3))
    arestas = arestas_h + arestas_v

    # Produto das quatro probabilidades = probabilidade de ocupaÃ§Ã£o simultanea de um bloco 2x2 de pixels
    faces = (p[:, :, :-1, :-1] * p[:, :, :-1, 1:] * p[:, :, 1:, :-1] * p[:, :, 1:, 1:]).sum(dim=(1, 2, 3))

    return vertices - arestas + faces


def caracteristica_euler_suave_8(imagem, temperatura=TEMPERATURA_PADRAO):
    """
    Caracteristica de Euler diferenciavel mas para vizinhanca de 8.

    Cada pixel ocupado Ã© um quadrado fechado:
        F = pixels ocupados
        E = arestas da grade tocadas por algum pixel ocupado
        V = cantos da grade tocados por algum pixel ocupado
        chi = V - E + F
    """
   
    # Intensidade dos pixels --> valores de 0 a 1
    p = torch.sigmoid((imagem - 0.5) / temperatura)

    # Soma as probabilidades de ocupaaoo de todos os pixels
    faces = p.sum(dim=(1, 2, 3))

    # Adiciona uma borda de pixels zerados ao redor da imagem
    pp = F.pad(p, (1, 1, 1, 1), value=0)

    a = pp[:, :, :-1, :-1]
    b = pp[:, :, :-1, 1:]
    c = pp[:, :, 1:, :-1]
    d = pp[:, :, 1:, 1:]

    # Probabilidade de pelo menos um dos quatro quadrados adjacentes esta ocupado
    vertices = (1.0 - (1.0 - a) * (1.0 - b) * (1.0 - c) * (1.0 - d)).sum(dim=(1, 2, 3))

    # Arestas horizontais e verticais tocadas por pelo menos um pixel ocupado
    ph = pp[:, :, :, 1:-1]
    arestas_h = (1.0 - (1.0 - ph[:, :, :-1, :]) * (1.0 - ph[:, :, 1:, :])).sum(dim=(1, 2, 3))

    pv = pp[:, :, 1:-1, :]
    arestas_v = (1.0 - (1.0 - pv[:, :, :, :-1]) * (1.0 - pv[:, :, :, 1:])).sum(dim=(1, 2, 3))

    return vertices - arestas_h - arestas_v + faces


# -----------------------------------------------------------
# Danos artificiais

def gerar_dano(imagem, rng=None, gerador=None):

    if rng is None:
        rng = random

    danificada = imagem.clone()

    _, altura, largura = danificada.shape

    # --------------------------------------------------------
    # Regioes retangulares sorteadas apagadas

    quantidade = rng.randint(1, 4)

    for _ in range(quantidade):

        h = rng.randint(8, 35)
        w = rng.randint(8, 45)
        y = rng.randint(0, altura - h)
        x = rng.randint(0, largura - w)

        valor = rng.choice([0.0, 1.0])

        danificada[:, y:y+h, x:x+w] = valor

    # --------------------------------------------------------
    # Riscos artificiais gerados 

    quantidade_riscos = rng.randint(1, 5)

    for _ in range(quantidade_riscos):

        x1 = rng.randint(0, largura - 1)
        y1 = rng.randint(0, altura - 1)
        x2 = rng.randint(0, largura - 1)
        y2 = rng.randint(0, altura - 1)

        espessura = rng.randint(1, 4)

        comprimento = max(abs(x2 - x1), abs(y2 - y1), 1)

        xs = torch.linspace(x1, x2, comprimento).long()
        ys = torch.linspace(y1, y2, comprimento).long()

        for x, y in zip(xs, ys):

            x = int(x)
            y = int(y)
            
            y_min = max(0, y - espessura)
            y_max = min(altura, y + espessura + 1)
            x_min = max(0, x - espessura)
            x_max = min(largura, x + espessura + 1)

            danificada[:, y_min:y_max, x_min:x_max] = 0.0

    # --------------------------------------------------------
    # Ruido gaussiano com media zero e desvio-padrao 0,08

    if rng.random() < 0.5:

        if gerador is None:
            ruido = torch.randn_like(danificada) * 0.08
        else:
            ruido = torch.randn(danificada.shape, generator=gerador) * 0.08

        danificada = torch.clamp(danificada + ruido, 0, 1)

    return danificada


# -----------------------------------------------------------
# Dataset

# Procura arquivos de imagem nas extensoes indicadas, organiza os caminhos em ordem alfabetica e verifica se a pasta contem imagens
def listar_imagens(pasta):

    pasta = Path(pasta)
    arquivos = []

    for extensao in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        arquivos.extend(pasta.glob(extensao))

    arquivos = sorted(arquivos)

    if len(arquivos) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em {pasta}")

    return arquivos

# Divide em treino e validacao
def dividir_arquivos(pasta, fracao_treino=0.85, seed=SEED):

    arquivos = listar_imagens(pasta)

    n_treino = int(fracao_treino * len(arquivos))

    gerador = torch.Generator().manual_seed(seed)
    ordem = torch.randperm(len(arquivos), generator=gerador).tolist()

    treino = [arquivos[i] for i in ordem[:n_treino]]
    validacao = [arquivos[i] for i in ordem[n_treino:]]

    return treino, validacao


# Abre a imagem, converte para RGB e aplica as transformacoes definidas
def carregar_imagem(caminho):
    return transformacao(Image.open(caminho).convert("RGB"))

# Fornece os pares de imagens ao autoencoder
class DatasetReconstrucao(Dataset):

    def __init__(self, arquivos, transform=None, danos_fixos=False, semente=SEED_VALIDACAO):

        self.arquivos = list(arquivos)
        self.transform = transform
        self.danos_fixos = danos_fixos

        if len(self.arquivos) == 0:
            raise RuntimeError("Lista de imagens vazia.")

        if danos_fixos:
            self.limpas = []
            self.danificadas = []

            for i in range(len(self.arquivos)):
                limpa = self._carregar(i)
                dano = gerar_dano(limpa, rng=random.Random(semente + i), gerador=torch.Generator().manual_seed(semente + i))
                self.limpas.append(limpa)
                self.danificadas.append(dano)

    def _carregar(self, indice):
        imagem = Image.open(self.arquivos[indice]).convert("RGB")
        if self.transform:
            imagem = self.transform(imagem)
        return imagem

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, indice):
        if self.danos_fixos:
            return self.danificadas[indice], self.limpas[indice]
        limpa = self._carregar(indice)
        return gerar_dano(limpa), limpa


# -----------------------------------------------------------
# Autoencoder convolucional

class AutoencoderCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()

        )

    def forward(self, x):

        return self.decoder(self.encoder(x))


# -----------------------------------------------------------
# Perda, calibracao de lambda e treinamento

# Perda com euler --> MSE + lambda * media(|chi(predicao) - chi(alvo)|)
def perda_total(predicao, alvo, lambda_euler, temperatura=TEMPERATURA_PADRAO):

    perda_mse = F.mse_loss(predicao, alvo)

    if lambda_euler == 0:
        with torch.no_grad():
            perda_euler = torch.mean(torch.abs(caracteristica_euler_suave_4(predicao, temperatura) - caracteristica_euler_suave_4(alvo, temperatura)))
        return perda_mse, perda_mse, perda_euler

    perda_euler = torch.mean(torch.abs(caracteristica_euler_suave_4(predicao, temperatura) - caracteristica_euler_suave_4(alvo, temperatura)))

    return perda_mse + lambda_euler * perda_euler, perda_mse, perda_euler

# Escolhe lambda para que o termo de Euler tenha, na escala inicial dos erros (imagem danificada vs limpa), um peso igual a `peso_relativo` vezes o do MSE
def calibrar_lambda(loader, peso_relativo=0.1, n_lotes=5, temperatura=TEMPERATURA_PADRAO):
   
    mses, eulers = [], []

    with torch.no_grad():

        for k, (danificada, limpa) in enumerate(loader):
            if k >= n_lotes:
                break

            mses.append(F.mse_loss(danificada, limpa).item())

            eulers.append(torch.mean(torch.abs(caracteristica_euler_suave_4(danificada, temperatura) - caracteristica_euler_suave_4(limpa, temperatura))).item())

    mse_medio = float(np.mean(mses))
    euler_medio = float(np.mean(eulers))

    return peso_relativo * mse_medio / max(euler_medio, 1e-8), mse_medio, euler_medio


# Treinamento
def treinar(
    modelo, loader_treino, loader_validacao,
    lambda_euler=0.0, epocas=100,
    taxa_aprendizado=1e-3,
    dispositivo="cpu",
    caminho_modelo=None,
    temperatura=TEMPERATURA_PADRAO,
    weight_decay=0.0,
    patience=10
):

    caminho_modelo = Path(caminho_modelo)
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)

    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa_aprendizado, weight_decay=weight_decay)

    historico = {"lambda_euler": float(lambda_euler), "treino": [], "validacao": [], "mse_validacao": [], "euler_validacao": []}

    melhor = float("inf")
    epocas_sem_melhora = 0

    for epoca in range(epocas):

        inicio = time.time()

        # ----------------------------------------------------
        # TREINO

        modelo.train()

        soma, n = 0.0, 0

        for entrada, alvo in loader_treino:

            entrada = entrada.to(dispositivo)
            alvo = alvo.to(dispositivo)
            
            otimizador.zero_grad()
            
            predicao = modelo(entrada)

            perda, _, _ = perda_total(predicao, alvo, lambda_euler, temperatura)

            perda.backward()
            otimizador.step()

            soma += perda.item() * len(entrada)
            n += len(entrada)

        perda_treino = soma / n

        # ----------------------------------------------------
        # Validacao (danos fixos -> curva comparavel entre epocas)

        modelo.eval()

        soma, soma_mse, soma_euler, n = 0.0, 0.0, 0.0, 0

        with torch.no_grad():

            for entrada, alvo in loader_validacao:

                entrada = entrada.to(dispositivo)
                alvo = alvo.to(dispositivo)

                predicao = modelo(entrada)

                perda, mse, euler = perda_total(predicao, alvo, lambda_euler, temperatura)

                soma += perda.item() * len(entrada)
                soma_mse += mse.item() * len(entrada)
                soma_euler += euler.item() * len(entrada)
                n += len(entrada)

        perda_val = soma / n

        historico["treino"].append(perda_treino)
        historico["validacao"].append(perda_val)
        historico["mse_validacao"].append(soma_mse / n)
        historico["euler_validacao"].append(soma_euler / n)

        marca = ""

        if perda_val < melhor:
            melhor = perda_val
            epocas_sem_melhora = 0
            torch.save(modelo.state_dict(), caminho_modelo)
            marca = " *"
        else:
            epocas_sem_melhora += 1

        print(f"Epoca {epoca+1:03d}/{epocas} | Treino: {perda_treino:.5f} | Validacao: {perda_val:.5f} | MSE val: {soma_mse / n:.5f} | |dChi| val: {soma_euler / n:.2f} | {time.time() - inicio:.0f}s{marca}")

        if epocas_sem_melhora >= patience:
            print(f"Early stopping na epoca {epoca + 1}")
            break

    # Restaura a melhor epoca
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo, weights_only=True))

    caminho_json = caminho_modelo.with_name(caminho_modelo.stem + "_historico.json")

    with open(caminho_json, "w") as arquivo:
        json.dump(historico, arquivo)

    print(f"\nMelhor validacao: {melhor:.5f}")
    print("Modelo salvo em:", caminho_modelo)

    return historico
