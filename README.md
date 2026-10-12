# pima-baseline-mlflow

Código de experimentación para un modelo base de clasificación sobre el dataset
**Pima Indians Diabetes**. Cada ejecución del script registra una corrida en
MLflow Tracking con sus hiperparámetros, sus métricas, los artefactos de
evaluación y el modelo entrenado.

El problema es binario: predecir si una paciente fue diagnosticada con diabetes
a partir de ocho mediciones clínicas. El modelo es una regresión logística, y el
objetivo no es el mejor desempeño posible sino un punto de partida reproducible.

## Estructura

```
.
├── train.py            script de entrenamiento parametrizado
├── requirements.txt    versiones con las que se produjeron las corridas
├── data/
│   └── pima.csv        768 registros, 8 predictores y la variable objetivo
└── notebooks/
    └── exploracion.ipynb   análisis exploratorio, sin lógica de modelado
```

## Requisitos

Python 3.11 o superior. El entorno se crea con [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Los comandos se ejecutan desde la raíz del repositorio: las rutas por defecto
(`data/pima.csv`, `artefactos/`) son relativas a ella.

Para abrir el notebook de exploración:

```bash
jupyter notebook notebooks/exploracion.ipynb
```

Para consultar las corridas, abrir `http://localhost:5000` y entrar al experimento
`baseline-pima-mlflow`.

El script registra contra un servidor de MLflow Tracking en
`http://localhost:5000`. Si ese puerto no responde, cae automáticamente a un
almacén local `sqlite:///mlflow_local.db` para que la ejecución no se bloquee.

## Ejecución

Con los valores por defecto:

```bash
python train.py
```

Cambiando hiperparámetros:

```bash
python train.py --C 0.01 --class-weight none --imputacion mean --semilla 10
```

Todos los argumentos tienen valor por defecto y ayuda descriptiva:

```bash
python train.py --help
```

| Argumento | Por defecto | Qué controla |
| --- | --- | --- |
| `--C` | `1.0` | Fuerza de la regularización, en forma inversa. `inf` entrena sin penalización |
| `--l1-ratio` | `0.0` | Tipo de penalización: `0.0` es L2 pura, `1.0` es L1 pura, intermedio es elastic net |
| `--solver` | `lbfgs` | Algoritmo de optimización. Solo `saga` admite `--l1-ratio` distinto de cero |
| `--class-weight` | `balanced` | Ponderación de clases ante el desbalance |
| `--imputacion` | `median` | Estrategia para rellenar faltantes |
| `--max-iter` | `1000` | Máximo de iteraciones del optimizador |
| `--tol` | `1e-4` | Tolerancia del criterio de paro |
| `--sin-intercepto` | desactivado | Entrena sin término independiente |
| `--test-size` | `0.25` | Proporción reservada para prueba |
| `--semilla` | `10` | Semilla aleatoria |
| `--datos` | `data/pima.csv` | Ruta al archivo de datos |
| `--experimento` | `baseline-pima-mlflow` | Experimento de MLflow |
| `--run-name` | derivado | Nombre de la corrida |
| `--tracking-uri` | `http://localhost:5000` | Servidor de MLflow |

### Nota sobre la penalización

El argumento `penalty` de `LogisticRegression` quedó obsoleto en scikit-learn 1.8 y
desaparece en la 1.10. El tipo de penalización se especifica ahora con `l1_ratio`,
y la ausencia de penalización con `C=inf`. Este script sigue la interfaz nueva.

El script valida las combinaciones antes de entrenar: pedir `--l1-ratio 1.0` con un
solver que no sea `saga` aborta con un mensaje que lo explica, en lugar de fallar
dentro de scikit-learn.

### Nota sobre la semilla

`--semilla` gobierna la partición entrenamiento/prueba, que es donde realmente se
decide el resultado. También se pasa a `LogisticRegression`, pero ahí solo tiene
efecto con los solvers que barajan datos (`sag`, `saga`, `liblinear`); con `lbfgs`
el ajuste ya es determinista. Se pasa de todas formas para que cambiar de solver no
introduzca una fuente de variación sin control.

## Las corridas comparadas

```bash
python train.py --run-name base
python train.py --C 0.05 --l1-ratio 1.0 --solver saga --max-iter 5000 --run-name lasso
python train.py --C 0.05 --l1-ratio 0.5 --solver saga --max-iter 5000 --run-name elastic-net
python train.py --C 0.05 --run-name ridge-fuerte
python train.py --C inf  --run-name sin-penalizacion
python train.py --class-weight none --run-name sin-ponderar
python train.py --semilla 42 --run-name semilla-42
```

Las siete quedan agrupadas en el experimento `baseline-pima-mlflow` y se
contrastan desde la vista **Compare** de la interfaz.

Las cinco primeras recorren el eje de la regularización, desde L1 pura hasta la
ausencia de penalización. La métrica `coeficientes_no_cero` hace visible el efecto:
con L1 y `C=0.05` el modelo descarta tres de las ocho variables.

La última no cambia ningún hiperparámetro, solo la semilla. Ilustra cuánto puede
moverse una métrica por la partición de los datos. Una sola semilla alternativa no
basta para medir esa variación; para distinguir con rigor una diferencia real de
una de muestreo haría falta validación cruzada o repetir cada configuración con
varias semillas.

## Qué se registra en cada corrida

- **Parámetros:** los hiperparámetros del modelo, el tamaño de los conjuntos y la semilla.
- **Métricas:** `accuracy`, `roc_auc`, `f1`, `recall`, `precision`, la exactitud de la clase mayoritaria como vara de comparación, los cuatro valores de la matriz de confusión y el número de coeficientes distintos de cero.
- **Artefactos:** matriz de confusión y curva ROC, reporte de clasificación, métricas en JSON, coeficientes con su razón de momios, y la huella del entorno de ejecución. Como los predictores se estandarizan, la razón de momios corresponde a un aumento de una desviación estándar de la variable, no de una unidad de su escala original.
- **Modelo:** el pipeline completo con `mlflow.sklearn.log_model`, con firma de entrada inferida y ejemplo de entrada.

## Reproducibilidad

La semilla se pasa a la partición de los datos y al optimizador de la regresión
logística, así que dos ejecuciones con los mismos argumentos producen las mismas
métricas. El archivo `requirements.txt` declara las versiones exactas, y cada
corrida registra además un artefacto `entorno.txt` con la versión de Python, la
plataforma y las versiones de las bibliotecas usadas.

## Decisión de limpieza

En `plas`, `pres`, `skin`, `test` y `mass` el valor cero es clínicamente
imposible: corresponde a un dato faltante codificado como cero. El script los
imputa con `SimpleImputer(missing_values=0)` aplicado únicamente a esas cinco
columnas. Los ceros de `preg` se conservan porque son legítimos e indican que no
hubo embarazos previos.

La limpieza y la imputación viven **dentro** del pipeline. Así la mediana se
calcula únicamente con los datos de entrenamiento y la misma transformación se
aplica después al conjunto de prueba, sin fuga de información.

## Datos

Pima Indians Diabetes Database, del National Institute of Diabetes and Digestive
and Kidney Diseases. 768 registros de mujeres de herencia Pima mayores de
veintiún años.
