#GAN CON REPO DE QUICK DRAW
## Keldryn Vannesa Sanchez Castro 

### Que es ? 
Entrenar una GAN (red generadora + red discriminadora) con una categoría del dataset QuickDraw (en mi ejemplo cat = gato) y generar dibujos nuevos.

### Que hice y como correrlo en powerShell? 

- Investigar los tres links de documentacion en el Caampus Virtual 
    https://developers.google.com/machine-learning/gan/gan_structure?hl=es-419
    https://poloclub.github.io/ganlab/
     
     y el repo 
     https://quickdraw.withgoogle.com/data

- Crear el proyecto con *uv init ganQuickDrawProyect* 
fijando python 3.12 y segun las guias usar *torch, numpy y matplotlib* era conveniente entonces se incluyen al proyecto  *uv add torch numpy matplotlib*

- Cree la clase *main.py* donde esta todo el codigo principal, mas adelante lo especifico un poco mas

- Se corre con * uv run main.py* PERO eso si se qyiere usar los valores por defecto (Al principio puse 30 epocas con 20000 imagenes pero se demoro un siglo) Actualmente estan en 15 epocas y 5000 imagenes

SI NO, SE PUEDEN PERSONALIZAR TANTO LAS EPOCAS Y LAS IMAGENES COMO LA CATEGORIA ASI 

*uv run main.py --categoria apple --epocas 10

--categoria:	Categoría de QuickDraw 
--epocas:	Cuántas veces pasa por todos los datos	(15 por defecto)
--n_imagenes	Cuántas imágenes de la categoría usa (5000 por defecto )


### Qué significa lo que hice 

1. Inicio

info Dispositivo: cpu
datos Categoria='cat'  imagenes=20000  forma=(20000, 1, 28, 28)
Dispositivo: donde se entrena (cpu en este caso)
Categoria: el tipo de dibujo que se está usando.
imagenes: cuántos dibujos se cargaron.
forma (5000, 1, 28, 28): 5000 imágenes, 1 canal (blanco y negro).

2. Resumen de las redes
GENERADOR
  proyeccion.0.weight   forma=(6272, 100)   parametros= 627,200
GENERADOR 
  TOTAL de parametros entrenables: 778,305
DISCRIMINADOR
  TOTAL de parametros entrenables: 138,817

Muestra cada capa de cada red, el tamaño de sus pesos (forma) y cuántos números aprende (parametros). El TOTAL es la cantidad de valores que la red ajusta durante el entrenamiento.

Generador: recibe ruido (100 números al azar) y crea una imagen.
Discriminador: recibe una imagen y dice si es real o falsa.

3. Una línea por época
epoca   1/30 perdida_D=0.6267  perdida_G=1.6994  D(x)=0.790  D(G(z))=0.292
Valor	Significado
epoca 1/30	Vuelta de entrenamiento actual / total
perdida_D	Error del discriminador. Mas bajo = distingue mejor real de falso
perdida_G	Error del generador. Mas bajo = engaña mejor al discriminador
D(x)	Que tan "real" cree el discriminador que son las imagenes reales (de 0 a 1)
D(G(z))	Que tan "real" cree que son las imagenes falsas (de 0 a 1)

Osea:

Al principio el discriminador gana: D(x) alto (cerca de 1) y D(G(z)) bajo (cerca de 0).
Cuando la GAN va bien, ambos se acercan a 0.5: el discriminador ya no puede distinguir lo real de lo falso.
Cuando el equilibrio es perfecto, perdida_D ronda 1.39 y perdida_G ronda 0.69.
En una GAN las perdidas no bajan de forma suave como en otros modelos: suben y bajan porque las dos redes compiten. Lo importante es que ninguna gane por completo.
Si perdida_D se acerca a 0 y perdida_G se dispara, el discriminador esta ganando demasiado y el generador no aprende.
4. Valores finales de los pesos
G.proyeccion.0.weight   media=+0.0000  std=0.0577
G.convs.1.weight        media=+1.0004  std=0.0060
D.clasificador.weight   media=-0.0011  std=0.0079

Para cada capa muestra la media y la desviacion estandar (std) de sus pesos. Sirve para ver los valores internos de cada red. G = generador, D = discriminador.

Nota: las capas con media cercana a 1.0 (como convs.1.weight) son capas de normalizacion (BatchNorm); es normal que valgan cerca de 1.

5. Final
listo Resultados guardados en: resultados/cat/

Indica que terminó y dónde quedaron los archivos.

Archivos de resultados (carpeta resultados categoria)
Archivo	Que contiene
resultado_final.png	Fila de dibujos reales y fila de dibujos generados, con la categoria en el titulo
perdidas.png	
Si algo se petateo puede ser : 
No descargar el dataset: bajar cat.npy desde https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy, crear una carpeta datos dentro del proyecto pa ponerlo ahi.
Falló la instalacion de torch:  hacer *uv python pin 3.12* para fijar python y volver a correr *uv add torch numpy matplotlib*

Listos Bais. <3