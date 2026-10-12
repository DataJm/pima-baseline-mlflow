"""
Entrenamiento del baseline de clasificacion sobre el dataset Pima Indians Diabetes.

Cada ejecucion registra una corrida en MLflow Tracking con sus hiperparametros,
sus metricas, los artefactos de evaluacion y el modelo entrenado.

Uso:
    python train.py
    python train.py --C 0.01 --class-weight none --run-name regularizacion-fuerte

"""

import argparse
import json
import platform
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Las cinco variables clinicas donde el valor cero es imposible y por lo tanto
# corresponde a un dato faltante codificado como cero. 'preg' queda fuera a
# proposito: sus ceros son legitimos, indican que no hubo embarazos previos.
COLUMNAS_CERO_IMPOSIBLE = ["plas", "pres", "skin", "test", "mass"]
OBJETIVO = "class"

def leer_argumentos(argv=None):
    """
    Define la interfaz de linea de comandos y devuelve los argumentos.
    """
    analizador = argparse.ArgumentParser(
        description="Entrena y registra en MLflow un baseline de regresion logistica "
                    "sobre el dataset Pima Indians Diabetes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--C", type=float, default=1.0,
                            help="Inverso de la fuerza de regularizacion. Valores pequenos regularizan mas. "
                                 "Usar 'inf' para una regresion logistica sin penalizacion.")
    analizador.add_argument("--l1-ratio", type=float, default=0.0,
                            help="Mezcla de la penalizacion: 0.0 es L2 pura, 1.0 es L1 pura, y cualquier "
                                 "valor intermedio es elastic net. Solo el solver 'saga' admite valores "
                                 "distintos de 0.0. Desde scikit-learn 1.8 este argumento sustituye a "
                                 "'penalty', que quedo obsoleto.")
    analizador.add_argument("--solver", type=str, default="lbfgs",
                            choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                            help="Algoritmo de optimizacion. 'saga' es el unico que admite L1 y elastic net.")
    analizador.add_argument("--class-weight", type=str, default="balanced",
                            choices=["balanced", "none"],
                            help="Ponderacion de clases. 'balanced' compensa el desbalance del dataset.")
    analizador.add_argument("--imputacion", type=str, default="median",
                            choices=["median", "mean", "most_frequent"],
                            help="Estrategia con la que se rellenan los faltantes.")
    analizador.add_argument("--max-iter", type=int, default=1000,
                            help="Maximo de iteraciones del optimizador.")
    analizador.add_argument("--tol", type=float, default=1e-4,
                            help="Tolerancia del criterio de paro del optimizador.")
    analizador.add_argument("--sin-intercepto", action="store_true",
                            help="Entrena sin termino independiente.")
    analizador.add_argument("--test-size", type=float, default=0.25,
                            help="Proporcion del conjunto reservada para prueba.")
    analizador.add_argument("--semilla", type=int, default=10,
                            help="Semilla aleatoria. Fija la particion entrenamiento/prueba y, en los "
                                 "solvers que barajan datos ('sag', 'saga', 'liblinear'), tambien el "
                                 "optimizador. En 'lbfgs' el ajuste ya es determinista.")
    analizador.add_argument("--datos", type=Path, default=Path("data/pima.csv"),
                            help="Ruta al archivo de datos.")
    analizador.add_argument("--experimento", type=str, default="baseline-pima-mlflow",
                            help="Nombre del experimento de MLflow que agrupa las corridas.")
    analizador.add_argument("--run-name", type=str, default=None,
                            help="Nombre de la corrida. Si se omite se arma con los hiperparametros.")
    analizador.add_argument("--tracking-uri", type=str, default="http://localhost:5000",
                            help="Servidor de MLflow Tracking.")
    args = analizador.parse_args(argv)
    if args.l1_ratio != 0.0 and args.solver != "saga":
        analizador.error(f"--l1-ratio {args.l1_ratio} exige --solver saga. "
                         f"El solver '{args.solver}' solo admite penalizacion L2 (--l1-ratio 0.0).")
    if not 0.0 <= args.l1_ratio <= 1.0:
        analizador.error("--l1-ratio debe estar entre 0.0 y 1.0.")
    if args.C <= 0:
        analizador.error("--C debe ser positivo.")
    if not 0.0 < args.test_size < 1.0:
        analizador.error("--test-size debe estar entre 0 y 1, sin incluirlos.")
    if args.max_iter < 1:
        analizador.error("--max-iter debe ser un entero positivo.")
    if args.tol <= 0:
        analizador.error("--tol debe ser positivo.")
    return args

def cargar_datos(ruta):
    """
    Lee el archivo de datos y separa predictores de variable objetivo.
    """
    if not ruta.exists():
        sys.exit(f"No se encontro el archivo de datos: {ruta}")
    datos = pd.read_csv(ruta)
    faltan = {OBJETIVO}.difference(datos.columns)
    if faltan:
        sys.exit(f"Al archivo le falta la columna objetivo: {faltan}")
    return datos.drop(columns=[OBJETIVO]), datos[OBJETIVO]

def construir_pipeline(args):
    """
    Arma el pipeline completo: limpieza, imputacion, escalado y clasificador.

    La limpieza y la imputacion viven dentro del pipeline a proposito. Asi la
    mediana se calcula unicamente con los datos de entrenamiento y la misma
    transformacion se aplica despues a prueba, sin fuga de informacion.
    """
    # SimpleImputer con missing_values=0 trata el cero como faltante sin pasar por
    # NaN, y solo en las columnas donde el cero es imposible. El resto pasa intacto.
    limpieza = ColumnTransformer(
        transformers=[
            ("ceros_imposibles",
             SimpleImputer(missing_values=0, strategy=args.imputacion),
             COLUMNAS_CERO_IMPOSIBLE),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("limpieza", limpieza),
        ("escalado", StandardScaler()),
        ("clasificador", LogisticRegression(
            C=args.C,
            l1_ratio=args.l1_ratio,
            solver=args.solver,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            max_iter=args.max_iter,
            tol=args.tol,
            fit_intercept=not args.sin_intercepto,
            random_state=args.semilla,
        )),
    ])


def evaluar(modelo, X_prueba, y_prueba, vara):
    """
    Calcula las metricas del conjunto de prueba.
    """
    prediccion = modelo.predict(X_prueba)
    probabilidad = modelo.predict_proba(X_prueba)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_prueba, prediccion).ravel()
    metricas = {
        "accuracy": accuracy_score(y_prueba, prediccion),
        "roc_auc": roc_auc_score(y_prueba, probabilidad),
        "f1": f1_score(y_prueba, prediccion),
        "recall": recall_score(y_prueba, prediccion),
        "precision": precision_score(y_prueba, prediccion),
        "accuracy_vara": vara,
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    }
    return prediccion, probabilidad, metricas


def generar_artefactos(carpeta, y_prueba, prediccion, probabilidad, modelo, metricas, args):
    """
    Escribe en disco la evidencia de la corrida y devuelve las rutas.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []

    figura, (izq, der) = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_prueba, prediccion),
        display_labels=["Sin diabetes", "Con diabetes"],
    ).plot(ax=izq, cmap="Blues", colorbar=False, values_format="d")
    izq.set_title("Matriz de confusion")
    fpr, tpr, _ = roc_curve(y_prueba, probabilidad)
    der.plot(fpr, tpr, label=f"AUC = {metricas['roc_auc']:.3f}")
    der.plot([0, 1], [0, 1], "k--", label="Azar")
    der.set_xlabel("Tasa de falsos positivos")
    der.set_ylabel("Sensibilidad")
    der.set_title("Curva ROC")
    der.legend(loc="lower right")
    figura.tight_layout()
    ruta_figura = carpeta / "evaluacion.png"
    figura.savefig(ruta_figura, dpi=120)
    plt.close(figura)
    rutas.append(ruta_figura)

    ruta_reporte = carpeta / "reporte_clasificacion.txt"
    ruta_reporte.write_text(
        classification_report(y_prueba, prediccion,
                              target_names=["Sin diabetes", "Con diabetes"], digits=3),
        encoding="utf-8")
    rutas.append(ruta_reporte)

    ruta_metricas = carpeta / "metricas.json"
    ruta_metricas.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    rutas.append(ruta_metricas)

    coeficientes = pd.DataFrame({
        "variable": modelo.named_steps["limpieza"].get_feature_names_out(),
        "coeficiente": modelo.named_steps["clasificador"].coef_[0],
    })
    # Los predictores se estandarizan antes del clasificador, asi que cada
    # coeficiente mide el efecto de aumentar la variable en una desviacion
    # estandar, no en una unidad de su escala original.
    coeficientes["razon_de_momios_por_desv_est"] = np.exp(coeficientes["coeficiente"])
    ruta_coeficientes = carpeta / "coeficientes.csv"
    coeficientes.sort_values("coeficiente", key=abs, ascending=False).to_csv(ruta_coeficientes, index=False)
    rutas.append(ruta_coeficientes)

    ruta_entorno = carpeta / "entorno.txt"
    ruta_entorno.write_text("\n".join([
        f"python={platform.python_version()}",
        f"plataforma={platform.platform()}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
        f"scikit-learn={sklearn.__version__}",
        f"mlflow={mlflow.__version__}",
        f"semilla={args.semilla}",
    ]), encoding="utf-8")
    rutas.append(ruta_entorno)

    return rutas


def resolver_tracking(uri, segundos=3):
    """Devuelve el almacen a usar. Si el servidor no responde, cae a SQLite local.

    Sin esta comprobacion el cliente espera de forma indefinida cuando el
    contenedor no esta levantado.
    """
    if not uri.startswith("http"):
        return uri
    destino = urlparse(uri)
    try:
        puerto = destino.port or (443 if destino.scheme == "https" else 80)
        with socket.create_connection((destino.hostname, puerto), timeout=segundos):
            return uri
    except OSError:
        print(f"El servidor {uri} no responde. Se usa el almacen local sqlite:///mlflow_local.db")
        return "sqlite:///mlflow_local.db"


def nombrar_corrida(args):
    """Arma un nombre de corrida legible a partir de los hiperparametros."""
    if args.run_name:
        return args.run_name
    return f"C{args.C}-l1r{args.l1_ratio}-{args.solver}-{args.class_weight}-{args.imputacion}"


def main(argv=None):
    """Orquesta una corrida: carga, particion, entrenamiento, evaluacion y registro.

    Escribe los artefactos en ./artefactos, relativo al directorio de ejecucion,
    y registra parametros, metricas, artefactos y el pipeline completo en MLflow.
    `argv` permite invocar la funcion con argumentos explicitos; si es None se
    usan los de la linea de comandos.
    """
    args = leer_argumentos(argv)

    X, y = cargar_datos(args.datos)
    # Particion estratificada: con 65 % de negativos, una particion al azar puede
    # dejar a prueba con una proporcion de clases distinta a la de entrenamiento.
    X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
        X, y, test_size=args.test_size, random_state=args.semilla, stratify=y)

    modelo = construir_pipeline(args)
    modelo.fit(X_entrena, y_entrena)

    # La vara: un clasificador que siempre predice la clase mayoritaria. Cualquier
    # configuracion que no la supere en exactitud no aprendio nada util.
    vara = DummyClassifier(strategy="most_frequent")
    vara.fit(X_entrena, y_entrena)
    exactitud_vara = accuracy_score(y_prueba, vara.predict(X_prueba))

    prediccion, probabilidad, metricas = evaluar(modelo, X_prueba, y_prueba, exactitud_vara)
    # Cuantas variables sobreviven a la penalizacion. Con L1 o elastic net algunos
    # coeficientes se anulan, y esta metrica deja visible esa seleccion en la vista
    # de comparacion de MLflow.
    metricas["coeficientes_no_cero"] = int((modelo.named_steps["clasificador"].coef_[0] != 0).sum())
    artefactos = generar_artefactos(Path("artefactos"), y_prueba, prediccion,
                                    probabilidad, modelo, metricas, args)

    mlflow.set_tracking_uri(resolver_tracking(args.tracking_uri))
    mlflow.set_experiment(args.experimento)

    with mlflow.start_run(run_name=nombrar_corrida(args)) as corrida:
        mlflow.log_params({
            "modelo": "LogisticRegression",
            "C": args.C,
            "l1_ratio": args.l1_ratio,
            "penalizacion": ("ninguna" if np.isinf(args.C)
                             else "L2" if args.l1_ratio == 0.0
                             else "L1" if args.l1_ratio == 1.0
                             else "elastic-net"),
            "solver": args.solver,
            "tol": args.tol,
            "fit_intercept": not args.sin_intercepto,
            "class_weight": args.class_weight,
            "imputacion": args.imputacion,
            "max_iter": args.max_iter,
            "escalamiento": "StandardScaler",
            "test_size": args.test_size,
            "semilla": args.semilla,
            "n_entrenamiento": X_entrena.shape[0],
            "n_prueba": X_prueba.shape[0],
            "n_variables": X.shape[1],
        })
        mlflow.log_metrics(metricas)
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta), artifact_path="evaluacion")
        # Se registra el pipeline completo, no solo el clasificador: la limpieza, la
        # imputacion y el escalado viajan con el modelo, asi que quien lo cargue
        # puede pasarle datos crudos sin reproducir el preprocesamiento a mano.
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            # Se declara la firma en punto flotante: las columnas enteras no pueden
            # representar faltantes, y MLflow avisa de ello al inferir el esquema.
            signature=infer_signature(X_entrena.astype(float), modelo.predict(X_entrena)),
            input_example=X_entrena.head(3).astype(float),
            # numpy.dtype aparece dentro del ColumnTransformer serializado con skops.
            skops_trusted_types=["numpy.dtype"],
        )
        mlflow.set_tag("dataset", "pima-indians-diabetes")

        print(f"Corrida   : {nombrar_corrida(args)}")
        print(f"Run ID    : {corrida.info.run_id}")
        print(f"Experimento: {args.experimento}")
        print("-" * 46)
        for nombre in ["accuracy", "accuracy_vara", "roc_auc", "recall", "precision", "f1"]:
            print(f"{nombre:16s} {metricas[nombre]:.4f}")


if __name__ == "__main__":
    main()
