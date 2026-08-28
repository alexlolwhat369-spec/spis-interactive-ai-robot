# SPIS Interactive AI Robot

Primer modulo del robot: reconoce gestos con una camara y activa una reaccion.
El modelo se entrena con coordenadas de manos, no con fotos ni rasgos faciales.

Gestos entrenables:

- `wave`
- `thumbs_up`
- `peace`
- `stop`
- `heart` (dos manos)

El estado `none` se produce cuando no hay una mano detectada. Si una mano no se
parece suficientemente a los ejemplos de entrenamiento, el sistema muestra
`unknown` en vez de adivinar.

Durante la demo en vivo hay una puerta adicional: el gesto debe ser cercano a
las muestras entrenadas, tener suficientes votos del modelo y mantenerse estable
durante unos fotogramas. Esto evita que una mano casual active una reaccion.

## Arquitectura portable

```text
Camara -> MediaPipe Hands -> 21 puntos por mano -> modelo KNN -> reaccion del robot
```

El entrenamiento puede hacerse en una laptop. Despues se copian estos archivos
a la Raspberry Pi 5: `src/`, `requirements.txt`, `models/hand_landmarker.task`
y `model/gesture_knn.npz`.
La Pi ejecuta solo `src/live_demo.py`; no necesita las muestras originales.

## Instalacion

Usa Python 3.11 o superior. En Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`opencv-contrib-python` es la unica distribucion de OpenCV que debe estar
instalada en este entorno. Instalarla junto a `opencv-python` puede reemplazar
archivos del modulo `cv2`.

Despues descarga una vez el modelo oficial que encuentra las manos. Ese archivo
queda guardado localmente; la deteccion no necesita internet despues.

```bash
python src/setup_assets.py
```

## Privacidad

El programa guarda solamente el nombre del gesto y 126 coordenadas numericas de
las manos. No guarda fotogramas ni rostros. La deteccion se ejecuta en el
dispositivo. Sin embargo, la version actual de MediaPipe declara que puede
enviar metricas de rendimiento y uso; durante la prueba local se observo un
intento de conexion para esa telemetria. Para una feria sin aprobacion explicita
de los visitantes, descarga las dependencias y el modelo antes, y usa la
Raspberry Pi sin conexion a internet durante la demo.

En Raspberry Pi OS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Capturar muestras

Repite la captura para cada gesto. Usa buena luz y cambia un poco el angulo y
la distancia de la mano. `heart` requiere que ambas manos aparezcan completas.

```bash
python src/collect_samples.py --label wave --samples 180
python src/collect_samples.py --label thumbs_up --samples 180
python src/collect_samples.py --label peace --samples 180
python src/collect_samples.py --label stop --samples 180
python src/collect_samples.py --label heart --samples 220
```

En la ventana de camara, presiona `C` para guardar una muestra y `Q` para salir.
No se guardan fotogramas: `data/landmarks.csv` contiene solo numeros de los
puntos de las manos.

Para una captura mas comoda, agrega `--auto`. El programa toma ejemplos solo
cuando pasa un intervalo corto y tus manos cambiaron lo suficiente; asi evita
guardar muchas copias casi iguales de una misma postura.

```bash
python src/collect_samples.py --label thumbs_up --samples 180 --auto
```

La captura usa la apertura automatica original de OpenCV, la misma ruta usada
para la primera recoleccion que funciono. Si una nueva computadora o camara da
problemas, `src/diagnose_camera.py` ofrece una prueba tecnica sin guardar ni
mostrar video, pero no forma parte del flujo normal de captura.

## Entrenar y evaluar

```bash
python src/train.py
```

El comando crea:

- `model/gesture_knn.npz`: el modelo para la demo y Raspberry Pi.
- `reports/evaluation.json`: precision y metricas por gesto.
- `reports/confusion_matrix.csv`: matriz de confusion.

No presentes una sola cifra de precision. Muestra tambien la matriz de
confusion, cuantas muestras hay por clase y ejemplos de errores.

La ultima evaluacion local uso 969 muestras: `97.4%` de precision sobre 194
muestras reservadas. La puerta de demo conserva 181 de esas muestras correctas
y rechaza las posturas que no son suficientemente cercanas.

## Datos publicos pequenos (HaGRID)

No descargues las imagenes de HaGRID: cada clase de imagenes puede ocupar decenas
de GB. Si ya descargaste las anotaciones oficiales de HaGRID en una carpeta,
este importador extrae solamente hasta 200 vectores de manos para cada una de
estas clases: `like -> thumbs_up`, `peace`, `stop`, `hand_heart -> heart`.
No guarda imagenes, identificadores de usuario ni metadata personal.

```bash
python src/import_hagrid.py --annotations-dir RUTA_A_LAS_ANOTACIONES --max-per-class 200
python src/train.py --dataset data/hagrid_landmarks.csv
```

El resultado se guarda en `data/hagrid_landmarks.csv`; mantenlo separado de
`data/landmarks.csv`, que contiene tus capturas voluntarias. Para la demo final,
usa tambien muestras de tu propia camara, especialmente para `wave`, que es un
gesto de movimiento y no aparece como una clase equivalente en HaGRID.

## Probar en vivo

```bash
python src/live_demo.py
```

`heart` dibuja un corazon azul y activa la reaccion `heart`. La cara animada
abre una segunda ventana de 800x480, apropiada para una pantalla HDMI pequena.
La voz se activa solo si el sistema tiene `espeak-ng` o `espeak` instalado; la
integracion de microfono y reconocimiento de voz se validara con el hardware.

Para ver todas las reacciones sin usar la webcam, ejecuta:

```bash
python src/face_demo.py
```

La pantalla avanza automaticamente. Usa las flechas izquierda/derecha o `A`/`D`
para cambiar de reaccion y `Q` para cerrar.

## Conversacion, juego y musica

Puedes probar el cerebro del robot sin microfono ni Ollama:

```bash
python src/chat_console.py
```

Para que el robot lea sus respuestas con la voz local del equipo, agrega
`--speak`. En Windows usa el sintetizador incluido por el sistema; en Raspberry
Pi usa `espeak-ng` cuando este instalado. Esta prueba aun recibe texto escrito:
el microfono y la transcripcion se conectaran despues de validar el dispositivo.

```bash
python src/chat_console.py --speak
```

El modo local entiende saludos, bromas, elogios, insultos suaves, confusion,
interes, una peticion de musica, el juego de objetos y detenerse. Reacciona con
una cara feliz ante elogios, una cara molesta pero amable ante insultos, una
cara confundida cuando no entiende y estrellas en los ojos ante interes claro.
El juego de objetos elige preguntas por ganancia de informacion y acepta `yes`,
`probably`, `maybe`, `probably not` y `no`. Para usar un modelo ya descargado
por Ollama, ejecuta:

```bash
python src/chat_console.py --ollama-model NOMBRE_DEL_MODELO
```

Si el juego falla una adivinanza, el robot pregunta por el objeto correcto. Si
ese objeto ya esta en el catalogo, guarda anonimamente las respuestas de la
ronda en `data/game_trials.jsonl` para mejorar las preguntas despues de revisar
los datos. Si es un objeto nuevo, tambien pregunta por una caracteristica que
lo diferencia del objeto equivocado y guarda la sugerencia en
`data/pending_object_suggestions.jsonl` para revision humana antes de agregarla
al catalogo. El algoritmo no tiene un limite de objetos; el catalogo curado
actual contiene 60 objetos.

La aplicacion valida que Ollama devuelva una respuesta, reaccion y accion
permitidas. Si Ollama no esta disponible, vuelve al modo local. El juego usa
`data/object_catalog.json`; puedes ampliar la lista sin reentrenar ningun
modelo. La musica usa `assets/music/playlist.json`, pero debes colocar pistas
originales o con permiso de uso en esa carpeta.

### Pruebas humanas del juego

Una simulacion confirma que los datos son consistentes, pero no reemplaza una
persona respondiendo preguntas reales. Para una ronda de prueba, un tester
elige en privado uno de los objetos, responde las preguntas y anota solamente
las respuestas normalizadas, la adivinanza y una nota opcional sobre preguntas
confusas. No guarda audio, video ni nombres.

```bash
python src/game_trial.py --list-targets
python src/game_trial.py --target laptop --trial-id round-01
```

Los resultados locales quedan en `data/game_trials.jsonl`. Haz entre 10 y 15
rondas con objetos distintos antes de la feria y cambia las preguntas que las
personas marquen como confusas.

Cuando tengas suficientes rondas, primero revisa el entrenamiento sin cambiar
el robot:

```bash
python src/train_object_game.py --dry-run
```

El entrenamiento usa una parte de las rondas como evaluacion reservada. Solo
calibra una pregunta cuando tiene al menos ocho respuestas para objetos que si
tienen la caracteristica y ocho para objetos que no. Al ejecutarlo sin
`--dry-run`, guarda la calibracion en `data/object_game_calibration.json` y un
reporte en `reports/object_game_training.json`. El robot carga esa calibracion
automaticamente en la siguiente partida.

```bash
python src/train_object_game.py
```

### Revisar sugerencias de aprendizaje

Cuando el robot falla, su sugerencia se mantiene separada del catalogo. Primero
listala y revisala:

```bash
python src/review_object_suggestions.py
```

Para aprobarla, un adulto o el equipo debe elegir una categoria, escribir los
atributos y definir una pregunta clara. El comando no acepta una sugerencia sin
esa revision explicita:

```bash
python src/review_object_suggestions.py --approve 1 --category technology --attributes electronic,fits_hand,temperature_sensor --attribute temperature_sensor --question "Is it mainly used to measure temperature?"
```

Para descartar una sugerencia incorrecta:

```bash
python src/review_object_suggestions.py --reject 1
```

Para laptop y una primera prueba en Raspberry Pi 5, el modelo recomendado es
`llama3.2:1b`. Ollama descarga ese modelo una vez y despues atiende las
conversaciones en `localhost`; no es entrenamiento desde cero. La Raspberry Pi
necesita memoria suficiente y una prueba real de velocidad antes de usarlo en
la feria.

### Conversacion por microfono

La demo hablada usa el reconocimiento ingles ya instalado en Windows para la
laptop y Vosk localmente en Raspberry Pi. La voz predeterminada es Piper con
la voz neural local `en_US-lessac-medium`, mas suave y natural que la voz
clasica de Windows. El modelo de voz se descarga una vez y despues el audio no
va a una API. Con auriculares evitas que el microfono escuche los altavoces
durante la prueba de laptop.

```bash
python src/voice_demo.py --ollama-model llama3.2:1b
```

Como respaldo, puedes usar la voz clasica de Windows:

```bash
python src/voice_demo.py --tts windows
```

Para mostrar los microfonos disponibles y elegir uno concreto:

```bash
python src/voice_demo.py --list-microphones
python src/voice_demo.py --microphone 1
python src/voice_demo.py --recognizer vosk
```

Para la feria, usa el modo unificado: la camara sigue detectando los gestos y
la cara conserva `listening`, `thinking` y `speaking` durante la conversacion.
Manten presionada la barra espaciadora mientras hablas y sueltala para que el
robot procese la frase. Durante la conversacion los gestos no cambian la cara;
despues debes retirar las manos de la camara antes de hacer una seña nueva.
Fuera de la conversacion, `heart` permanece visible durante 1.5 segundos para
que no desaparezca si las manos salen de la camara.

```bash
python src/interactive_robot.py --ollama-model spis-robot --recognizer vosk --microphone 1
```

El modelo `spis-robot` es una configuracion local de `llama3.2:1b` con reglas y
ejemplos del proyecto, creada con `config/spis-robot.Modelfile`. No es un
entrenamiento de pesos desde cero; para un ajuste de pesos real necesitariamos
un conjunto amplio de dialogos voluntarios, etiquetados y evaluados.

## Copiar a Raspberry Pi

1. Copia la carpeta del proyecto a la Pi por USB, Git o red local.
2. Instala las dependencias y ejecuta `python src/setup_assets.py` en la Pi.
3. Conecta la camara oficial y prueba `python src/live_demo.py --camera 0`.
4. Si la camara usa otro indice, cambia `--camera` a `1` o `2`.

La primera prueba debe hacerse con una webcam USB o con la camara Raspberry
configurada como dispositivo V4L2. La integracion con `Picamera2` se agrega
cuando tengamos la Pi fisica, para evitar escribir codigo sin poder probar el
hardware real.
