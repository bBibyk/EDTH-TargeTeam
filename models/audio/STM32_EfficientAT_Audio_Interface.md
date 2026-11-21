# STM32 B-L475E-IOT01A → EfficientAT --- contrat technique

## Objectif

La STM32 doit transformer l'audio capté en **exactement la
représentation d'entrée utilisée par notre EfficientAT**.

``` text
Micro → audio mono 32 kHz → fenêtre 2 s → préaccentuation → STFT
→ puissance → 128 bandes Mel → ln → normalisation
→ float32 [1,1,128,200] → EfficientAT ONNX → 4 logits
```

## Entrée audio de référence

  Paramètre                    Valeur
  ------------------- ---------------
  Sample rate           **32 000 Hz**
  Canaux                 **1 (mono)**
  Fenêtre                 **2,000 s**
  Échantillons             **64 000**
  Type de référence       **float32**

Entrée du prétraitement : `float32[64000]`.

**Attention :** si le micro STM32 fournit du PCM entier, la conversion
PCM → float doit être validée contre le pipeline de référence. Ne pas
inventer une normalisation différente.

## 1. Préaccentuation

Noyau exact : `[-0.97, 1.0]`, convolution `valid`.

``` text
y[n] = x[n+1] - 0.97*x[n]
n = 0..63998
```

Entrée : 64 000 samples. Sortie : **63 999 samples**.

``` c
for (n = 0; n < 63999; n++)
    y[n] = x[n + 1] - 0.97f * x[n];
```

Ne pas ajouter un échantillon pour conserver artificiellement une
longueur de 64 000.

## 2. STFT

Paramètres exacts :

``` yaml
n_fft: 1024
win_length: 800
hop_length: 320
center: true
pad_mode: reflect
normalized: false
window: Hann
window_periodic: false
```

### Fenêtre Hann

Équivalent à `torch.hann_window(800, periodic=False)` :

``` text
w[n] = 0.5 * (1 - cos(2*pi*n/799))
n = 0..799
```

Le dénominateur est **799**, pas 800.

### Padding du signal

`center=True` avec `n_fft=1024` ajoute **512 samples de chaque côté** en
mode `reflect`.

``` text
63999 + 512 + 512 = 65023 samples
```

Ce n'est pas du zero-padding.

### Placement de la Hann dans les 1024 points FFT

La fenêtre active fait 800 points et est centrée dans 1024 :

``` text
(1024 - 800) / 2 = 112
```

Donc :

``` text
indices 0..111    : 0
indices 112..911  : Hann[0..799]
indices 912..1023 : 0
```

### Découpage temporel

``` text
hop = 320 samples = 10 ms à 32 kHz
Hann = 800 samples = 25 ms à 32 kHz
```

Le résultat attendu est **200 frames temporelles**.

Une FFT réelle 1024 donne **513 bins** : `1024/2 + 1`.

## 3. Spectre de puissance

Pour chaque coefficient complexe :

``` text
power = |X|² = Re² + Im²
```

Utiliser la **puissance**, pas simplement `|X|`.

La STFT de référence utilise `normalized=False` : ne pas introduire une
normalisation FFT supplémentaire.

Avant Mel, la représentation est conceptuellement :

``` text
[513 fréquences, 200 frames]
```

## 4. Banque Mel

Paramètres exacts :

``` yaml
sample_rate: 32000
n_fft: 1024
fft_bins: 513
n_mels: 128
fmin_hz: 0
fmax_hz: 15000
htk: true
norm: null
```

La matrice validée est fournie dans `mel_basis.npy`.

``` text
shape: [128,513]
dtype: float32
SHA256:
42bcd89766a85cd15ebaa640fe3627d2a7db949888cbb9d0613b4a4749befc54
```

**Recommandation : utiliser directement cette matrice plutôt que de
régénérer les filtres Mel sur STM32.**

Elle contient `128 × 513 = 65 664` coefficients.

Pour chaque frame :

``` text
mel[128] = MEL_BASIS[128x513] × power[513]
```

Résultat : **\[128,200\]**.

## 5. Logarithme

Pour chaque valeur Mel :

``` text
log_mel = ln(mel + 1e-5)
```

C'est le **logarithme naturel**.

Ne pas utiliser `log10`, `10*log10`, `20*log10` ou une conversion dB.

## 6. Normalisation finale

``` text
output = (log_mel + 4.5) / 5.0
```

Constantes exactes :

``` text
epsilon = 1e-5
offset  = 4.5
scale   = 5.0
```

## 7. Tensor attendu par le modèle

``` yaml
input_name: mel
shape: [1, 1, 128, 200]
dtype: float32
batch: 1
```

Dimensions :

``` text
[batch, channel, mel_frequency, time]
[  1,      1,          128,       200]
```

Le batch ONNX est **fixe à 1**.

Le tensor contient `25 600` float32, soit `102 400 octets` (\~100 KiB).

## 8. Sortie EfficientAT

``` yaml
output_name: logits
shape: [1,4]
dtype: float32
```

Ordre **exact** des classes :

``` text
0 = background
1 = vehicle
2 = drone
3 = footsteps
```

La sortie contient des **logits**, pas des probabilités.

Probabilités :

``` text
p = softmax(logits)
P(vehicle) = p[1]
```

Le seuil opérationnel `P(vehicle) >= threshold` n'est **pas encore
figé** : il doit être calibré sur les essais terrain.

## 9. Référence mathématique compacte

``` text
x : float32[64000]

1) preemphasis
   x1[n] = x[n+1] - 0.97*x[n]
   → 63999 samples

2) reflect padding
   512 à gauche + 512 à droite
   → 65023 samples

3) STFT
   n_fft=1024
   win_length=800
   hop=320
   Hann periodic=False
   normalized=False
   → 200 frames × 513 bins complexes

4) power
   P = Re² + Im²
   → [513,200]

5) Mel
   M = MEL_BASIS[128,513] @ P[513,200]
   → [128,200]

6) log
   L = ln(M + 1e-5)

7) normalize
   Z = (L + 4.5) / 5.0

8) reshape
   → float32[1,1,128,200]
```

## 10. Implémentation légère déjà validée

L'équivalent NumPy utilisé pour le déploiement est :

``` python
x = x[1:] - np.float32(0.97) * x[:-1]

x = np.pad(x, (512, 512), mode="reflect")

frames = np.lib.stride_tricks.sliding_window_view(x, 1024)[::320]

padded_window = np.zeros(1024, dtype=np.float32)
padded_window[112:912] = np.hanning(800).astype(np.float32)

frames = frames * padded_window[None, :]

spectrum = np.fft.rfft(frames, n=1024, axis=1)

power = (
    spectrum.real * spectrum.real
    + spectrum.imag * spectrum.imag
).astype(np.float32).T

melspec = MEL_BASIS @ power
melspec = np.log(melspec + np.float32(1e-5))
melspec = (melspec + np.float32(4.5)) / np.float32(5.0)

melspec = melspec[np.newaxis, np.newaxis, :, :]
```

Sortie : `float32[1,1,128,200]`.

## 11. Validation déjà obtenue

### Prétraitement léger vs PyTorch --- 136 fichiers

Jeu équilibré : 34 `background`, 34 `vehicle`, 34 `drone`, 34
`footsteps`.

``` text
shape errors               : 0
max absolute error global  : 3.4332275390625e-05
mean absolute error global : 2.314682427754821e-08
median max error           : 9.5367431640625e-07
P95 max error              : 7.987022399902344e-06
P99 max error              : 1.676082611083986e-05
```

### Pipeline complet

Comparaison :

``` text
audio → Mel PyTorch → EfficientAT PyTorch
vs
audio → Mel NumPy → EfficientAT ONNX
```

Sur 136 fichiers :

``` text
accord argmax       : 136/136 = 100 %
max logit error     : 6.008148193359375e-05
mean logit error    : 7.80251096276676e-06
P95 max logit error : 4.2557716369628906e-05
P99 max logit error : 5.364418029785159e-05
désaccords          : 0
```

Ces valeurs servent de référence pour tester la version STM32.

## 12. Validation obligatoire de l'implémentation STM32

Ne pas valider uniquement en regardant la classe finale.

Prendre **exactement le même buffer de 64 000 samples** :

``` text
                   même audio
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
     Python référence          STM32
           │                     │
           ▼                     ▼
 [1,1,128,200] float32   [1,1,128,200]
           └──────────┬──────────┘
                      ▼
             comparaison numérique
```

Comparer au minimum :

``` text
shape
max absolute error
mean absolute error
P95 absolute error
P99 absolute error
```

Ensuite injecter les deux tensors dans **le même ONNX** et comparer :

``` text
4 logits
argmax
P(vehicle)
```

### Si divergence, comparer étape par étape

``` text
1. audio float32
2. préaccentuation
3. reflect padding
4. coefficients Hann
5. frame FFT
6. Re/Im puis puissance
7. sortie Mel avant log
8. sortie après ln
9. sortie normalisée
10. tensor final [1,1,128,200]
```

## 13. Pièges à éviter

1.  sample rate autre que **32 kHz** ;
2.  mauvaise conversion PCM → float ;
3.  mauvais mixage mono ;
4.  préaccentuation qui conserve 64 000 samples ;
5.  padding autre que **reflect** ;
6.  Hann périodique au lieu de `periodic=False` ;
7.  Hann 800 mal centrée dans FFT 1024 ;
8.  FFT normalisée différemment ;
9.  magnitude au lieu de **Re² + Im²** ;
10. banque Mel régénérée différemment ;
11. `log10` au lieu de **ln** ;
12. oubli du `+1e-5` ;
13. oubli de `(x+4.5)/5.0` ;
14. ordre des dimensions différent de `[1,1,128,200]` ;
15. fixed-point/quantification avant validation float32.

## 14. Mémoire utile pour la STM32

Ordres de grandeur en float32 :

``` text
audio 64000             : 256 000 octets ≈ 250 KiB
Mel basis 128×513       : 262 656 octets ≈ 256.5 KiB
tensor final 128×200    : 102 400 octets ≈ 100 KiB
```

Il n'est pas nécessaire de conserver toutes les représentations
intermédiaires simultanément. Un traitement frame par frame / streaming
peut être nécessaire sur la STM32.

Toute optimisation mémoire ou fixed-point doit être faite **après**
obtention d'une version de référence numériquement compatible.

## 15. Fichiers fournis

Package Raspberry :

``` text
efficientat_audio_v1.onnx
efficientat_audio_v1.onnx.data
mel_basis.npy
inference_pi.py
requirements.txt
SHA256SUMS
```

Hashes :

``` text
efficientat_audio_v1.onnx
7fd42269e59d98dcb301ea631665e37a0fd7fb1b2dd590aef3980a1dbd65d808

efficientat_audio_v1.onnx.data
1fb8af330ec62ad9e17ceee374f599c12c39fb59fab61d10b2dc7a9bddcc5b96

mel_basis.npy
42bcd89766a85cd15ebaa640fe3627d2a7db949888cbb9d0613b4a4749befc54
```

Le fichier `.onnx.data` est obligatoire avec le `.onnx`.

## 16. Résumé machine-readable

``` yaml
audio:
  sample_rate_hz: 32000
  channels: 1
  window_seconds: 2.0
  samples_per_window: 64000
  reference_dtype: float32

preemphasis:
  kernel: [-0.97, 1.0]
  formula: "y[n] = x[n+1] - 0.97*x[n]"
  output_samples: 63999

stft:
  n_fft: 1024
  win_length: 800
  hop_length: 320
  center: true
  pad_mode: reflect
  pad_each_side: 512
  normalized: false
  window_type: hann
  window_periodic: false
  fft_bins: 513
  time_frames: 200
  spectrum: power
  power_formula: "real^2 + imag^2"

mel:
  n_mels: 128
  fmin_hz: 0
  fmax_hz: 15000
  htk: true
  norm: null
  matrix_file: mel_basis.npy
  matrix_shape: [128, 513]
  matrix_dtype: float32
  matrix_sha256: "42bcd89766a85cd15ebaa640fe3627d2a7db949888cbb9d0613b4a4749befc54"

postprocessing:
  logarithm: natural
  epsilon: 0.00001
  normalization: "(ln(mel + 1e-5) + 4.5) / 5.0"

model_input:
  name: mel
  shape: [1, 1, 128, 200]
  dtype: float32
  batch_size: 1

model_output:
  name: logits
  shape: [1, 4]
  dtype: float32
  class_order: [background, vehicle, drone, footsteps]
  vehicle_index: 1

decision:
  probability: "softmax(logits)"
  vehicle_probability: "softmax(logits)[1]"
  threshold: "TO_BE_CALIBRATED"
```

## Critère de réussite

La STM32 est compatible lorsque, pour les **mêmes buffers audio**, elle
produit un tensor `[1,1,128,200]` suffisamment proche de la référence
Python et que les deux tensors produisent des logits / décisions
cohérents dans le même modèle ONNX.

**Ne pas quantifier ou modifier le prétraitement avant d'avoir établi
cette parité.**
