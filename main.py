#Holis, esta va a ser la clase principal

#Me tocaba hacer esto 
#  1) La red neuronal GENERADORA      -> clase Generador
#  2) La red neuronal DISCRIMINADORA  -> clase Discriminador
#  3) Las FUNCIONES DE PERDIDA        -> ver bucle de entrenamiento
#  4) Los VALORES del generador y del discriminador (perdidas, D(x),
#     D(G(z)), estadisticas de los pesos de cada capa)
#  5) El RESULTADO de la GAN, indicando la CATEGORIA usada

# Usando el Dataset: QuickDraw (https://github.com/googlecreativelab/quickdraw-dataset)
# Usando numpy_bitmap

#Anton tiruliru:
#python gan_quickdraw.py --categoria cat --epocas 30
#python gan_quickdraw.py --categoria apple --epocas 30
#python gan_quickdraw.py --sintetico --epocas 2  

import argparse
import csv
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")  # para guardar imagenes mas facil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#1. Los parametrod

DIM_LATENTE = 100      # tamano que recibe el generador
LR = 2e-4              # aprendizaje (learning rate)
BETAS = (0.5, 0.999)   # Optimizador Adam 
TAM_BATCH = 128        # imagenes 
N_IMAGENES = 5000     # cuantas imagenes de esa categoria se usan


#2. Datos de QuickDraw

def cargar_quickdraw(categoria, n_imagenes, sintetico=False): #Descarga (si no existe) el .npy de la categoria y lo devuelve como escala o formita
    
    if sintetico:
        # Datos falsos garabatos para ver si funciona :p
        print("[datos] Usando datos SINTETICOS (just prueba).")
        yy, xx = np.mgrid[0:28, 0:28]
        imgs = []
        for _ in range(n_imagenes):
            cx, cy, r = np.random.randint(8, 20, 2).tolist() + [np.random.randint(3, 8)]
            imgs.append((((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2) * 255)
        datos = np.array(imgs, dtype=np.uint8).reshape(-1, 784)
    else:
        os.makedirs("datos", exist_ok=True)
        ruta = os.path.join("datos", f"{categoria}.npy")
        if not os.path.exists(ruta):
            # URL oficial del dataset de QuickaDraw 
            url = ("https://storage.googleapis.com/quickdraw_dataset/full/"
                   f"numpy_bitmap/{categoria}.npy")
            print(f"[datos] Descargando {url} ...")
            urllib.request.urlretrieve(url, ruta)
        datos = np.load(ruta)[:n_imagenes]  # forma (N, 784), uint8 0..255

    datos = datos.reshape(-1, 1, 28, 28).astype("float32")
    datos = (datos - 127.5) / 127.5  # [0,255]-> [-1,1] (coincide)
    print(f"[datos] Categoria='{categoria}'  imagenes={len(datos)}  forma={datos.shape}")
    return torch.from_numpy(datos)


#3.RED NEURONAL GENERADORA
class Generador(nn.Module):   # GENERADOR: recibe ruido aleatorio z (vector de 100 numeros) y lotransforma en una imagen.
                                # Su objetivo es que el discriminador crea que la imagen es real.
                                #Flujo de formas (shapes):z (100) -> Linear -> (128*7*7) -> reshape (128,7,7) -> ConvTranspose2d -> (64,14,14)
                                #-> ConvTranspose2d -> (1,28,28) -> Tanh  [-1,1]

    def __init__(self, dim_latente=DIM_LATENTE):
        super().__init__()
        # Compara l que ve con un mapa de coordenadas donde se van las formas por asi decirlo
        self.proyeccion = nn.Sequential(
            nn.Linear(dim_latente, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),   # estabiliza el entrenamiento
            nn.ReLU(True),
        )
        # Agranda la imagen
        self.convs = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 14x14 -> 28x28, 1 canal (escala de grises)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # salida en [-1,1], igual que los datos normalizados
        )

    def forward(self, z):
        x = self.proyeccion(z)
        x = x.view(-1, 128, 7, 7)
        return self.convs(x)

# 4. RED NEURONAL DISCRIMINADORA

class Discriminador(nn.Module): #DISCRIMINADOR: clasificador binario. Recibe una imagen y
                                #devuelve un numero: positivo = "real", negativo = "falsa".
                                #Grafico de formas que se van entendiendo: (1,28,28) -> Conv2d -> (64,14,14) -> Conv2d -> (128,7,7)-> Flatten -> Linear -> 1 logit
   

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),                                         # evita que D "gane" muy rapido
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.clasificador = nn.Linear(128 * 7 * 7, 1)  # 1 salida: real/falsa

    def forward(self, x):
        x = self.convs(x)
        return self.clasificador(x.flatten(1))


#5. UTILIDADES PARA "VER LOS VALORES" DE CADA RED
def resumen_red(nombre, red): #Imprime cada capa con su numero de pesos
    print(f"\n===== {nombre} =====")
    total = 0
    for n, p in red.named_parameters():
        print(f"  {n:<35} forma={str(tuple(p.shape)):<20} parametros={p.numel():>8,}")
        total += p.numel()
    print(f"  TOTAL de parametros entrenables: {total:,}")
    return total


def estadisticas_pesos(nombre, red): #Devuelve media y desviacion estandar de los pesos de cada capa (asi se ve como cambian los valores de la red durante el entrenamiento).
    filas = []
    for n, p in red.named_parameters():
        if "weight" in n:
            filas.append((f"{nombre}.{n}", p.mean().item(), p.std().item()))
    return filas


def guardar_muestras(gen, z_fijo, ruta, titulo, dispositivo): #Genera 16 imagenes con un ruido fijo y las guarda en una cuadricula 4x4."""
    gen.eval()
    with torch.no_grad():
        imgs = gen(z_fijo.to(dispositivo)).cpu().numpy()
    gen.train()
    fig, ejes = plt.subplots(4, 4, figsize=(5, 5.4))
    for ax, im in zip(ejes.flat, imgs):
        ax.imshow(im[0], cmap="gray_r", vmin=-1, vmax=1)  # trazo oscuro sobre fondo claro
        ax.axis("off")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


#6. Entrenamiento 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categoria", default="cat", help="categoria de QuickDraw (cat, apple, house...)")
    ap.add_argument("--epocas", type=int, default=10) # Solo puse 10 epocas porque tneia 30 y se demoro un siglo
    ap.add_argument("--sintetico", action="store_true", help="usar datos falsos (prueba)")
    ap.add_argument("--n_imagenes", type=int, default=N_IMAGENES)
    args = ap.parse_args()

    categoria = "circulos_sinteticos" if args.sintetico else args.categoria
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Dispositivo: {dispositivo}")

    salida = os.path.join("resultados", categoria)
    os.makedirs(salida, exist_ok=True)

    #Datos
    datos = cargar_quickdraw(categoria, args.n_imagenes, args.sintetico)
    loader = DataLoader(TensorDataset(datos), batch_size=TAM_BATCH, shuffle=True, drop_last=True)

    # AQUI SE CREA LA RED NEURONAL GENERADORA
    generador = Generador().to(dispositivo)

    #AQUI SE CREA LA RED NEURONAL DISCRIMINADORA
    discriminador = Discriminador().to(dispositivo)

    resumen_red("GENERADOR", generador)
    resumen_red("DISCRIMINADOR", discriminador)

    #AQUI SE DEFINE LA FUNCION DE PERDIDA
    perdida_bce = nn.BCEWithLogitsLoss()

    # Un optimizador (Adam) por cada red: cada una actualiza solo los pesos suyos.
    opt_g = torch.optim.Adam(generador.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(discriminador.parameters(), lr=LR, betas=BETAS)

    z_fijo = torch.randn(16, DIM_LATENTE)  # mismo ruido en todas las epocas para ver la evolucion
    historial = []  # aqui se guardan los valores de cada epoca

    for epoca in range(1, args.epocas + 1):
        sd, sg, sdx, sdgz, nb = 0.0, 0.0, 0.0, 0.0, 0
        for (reales,) in loader:
            reales = reales.to(dispositivo)
            n = reales.size(0)
            etiqueta_real = torch.ones(n, 1, device=dispositivo)   # y = 1
            etiqueta_falsa = torch.zeros(n, 1, device=dispositivo) # y = 0

            # PASO A: entrenar el discriminador
            #   Perdida_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
            #   -> debe dar alto a las reales y bajo a las falsas.
            z = torch.randn(n, DIM_LATENTE, device=dispositivo)
            falsas = generador(z)
            salida_real = discriminador(reales)
            salida_falsa = discriminador(falsas.detach())  
            perdida_d = (perdida_bce(salida_real, etiqueta_real)
                         + perdida_bce(salida_falsa, etiqueta_falsa))
            opt_d.zero_grad()
            perdida_d.backward()
            opt_d.step()

            #PASO B: entrenar el generador
            #   Perdida_G = BCE(D(G(z)), 1)   (version "no saturante")
            #   -> el generador quiere que D diga "real" a sus falsas.
            salida_falsa_g = discriminador(falsas)  # ahora SI pasa gradiente al generador
            perdida_g = perdida_bce(salida_falsa_g, etiqueta_real)
            opt_g.zero_grad()
            perdida_g.backward()
            opt_g.step()

            # Valores para revisar
            sd += perdida_d.item()
            sg += perdida_g.item()
            sdx += torch.sigmoid(salida_real).mean().item()    
            sdgz += torch.sigmoid(salida_falsa).mean().item()
            nb += 1 #Esto cuaenta cuantos lotes se procesan o si no queda en cero y se petatea (me pasó)

        fila = dict(epoca=epoca, perdida_D=sd / nb, perdida_G=sg / nb,
                    D_de_x=sdx / nb, D_de_Gz=sdgz / nb)
        historial.append(fila)
        print(f"[epoca {epoca:>3}/{args.epocas}] perdida_D={fila['perdida_D']:.4f}  "
              f"perdida_G={fila['perdida_G']:.4f}  D(x)={fila['D_de_x']:.3f}  "
              f"D(G(z))={fila['D_de_Gz']:.3f}")
        # Cuando la GAN: D(x) y D(G(z)) se acercan a 0.5 (D ya no lo identifica).

        if epoca == 1 or epoca % 5 == 0 or epoca == args.epocas:
            guardar_muestras(generador, z_fijo,
                             os.path.join(salida, f"muestras_epoca_{epoca:03d}.png"),
                             f"GAN - categoria: {categoria} - epoca {epoca}", dispositivo)

    #7. Guardar valores y resultado
    with open(os.path.join(salida, "valores_por_epoca.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=historial[0].keys())
        w.writeheader()
        w.writerows(historial)

    ep = [h["epoca"] for h in historial]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["perdida_D"] for h in historial], label="Perdida Discriminador")
    a1.plot(ep, [h["perdida_G"] for h in historial], label="Perdida Generador")
    a1.set_xlabel("epoca"); a1.set_ylabel("perdida (BCE)"); a1.set_title("Funciones de perdida"); a1.legend()
    a2.plot(ep, [h["D_de_x"] for h in historial], label="D(x) sobre imagenes reales")
    a2.plot(ep, [h["D_de_Gz"] for h in historial], label="D(G(z)) sobre imagenes falsas")
    a2.axhline(0.5, color="gray", ls="--", label="0.5 = D no distingue")
    a2.set_xlabel("epoca"); a2.set_title("Salida del discriminador"); a2.legend()
    fig.suptitle(f"Categoria: {categoria}")
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "perdidas.png"), dpi=110)
    plt.close(fig)

    print("\n===== VALORES FINALES DE LOS PESOS (media / desv. estandar por capa) =====")
    for nombre, red in (("G", generador), ("D", discriminador)):
        for capa, m, s in estadisticas_pesos(nombre, red):
            print(f"  {capa:<38} media={m:+.4f}  std={s:.4f}")

    # Resultado final: real vs generado, con categoria
    generador.eval()
    with torch.no_grad():
        falsas = generador(torch.randn(8, DIM_LATENTE, device=dispositivo)).cpu().numpy()
    reales = datos[:8].numpy()
    fig, ejes = plt.subplots(2, 8, figsize=(12, 3.6))
    for i in range(8):
        ejes[0, i].imshow(reales[i, 0], cmap="gray_r", vmin=-1, vmax=1); ejes[0, i].axis("off")
        ejes[1, i].imshow(falsas[i, 0], cmap="gray_r", vmin=-1, vmax=1); ejes[1, i].axis("off")
    ejes[0, 0].set_title("REALES (QuickDraw)", loc="left")
    ejes[1, 0].set_title("GENERADAS por la GAN", loc="left")
    fig.suptitle(f"Resultado final - categoria usada: '{categoria}'", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "resultado_final.png"), dpi=110)
    plt.close(fig)

    torch.save(generador.state_dict(), os.path.join(salida, "generador.pt"))
    torch.save(discriminador.state_dict(), os.path.join(salida, "discriminador.pt"))
    print(f"\n[listo] Resultados guardados en: {salida}/")


if __name__ == "__main__":
    main()