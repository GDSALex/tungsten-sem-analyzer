"""Evolutionary tuning of the classical pipeline (NSGA-II through Optuna).

The classical detector of :mod:`tungsten_analyzer.classical_pipeline` exposes forty
free parameters, listed in ``classical_pipeline.CFG``, and they interact strongly
enough that tuning them by hand is hopeless. This module searches that space with
the NSGA-II sampler provided by Optuna, using a population of twenty individuals
and a mutation probability of 0.1.

One individual is one complete parameter set. Evaluating it means running the whole
classical pipeline over every annotated micrograph and scoring the output against
the manual annotation. At this stage the annotation is collapsed into a single
class-agnostic damage mask, craters and cracks merged, because the aim is to tune
the detectors jointly rather than to tell the two morphologies apart. For a
predicted mask P and a reference mask G the score of one image is

    F = 0.3 * IoU + 0.7 * F1,

and the fitness handed back to the optimiser is the mean of F over the dataset. F1
carries the larger weight because, for a given number of missed pixels, it is the
less punitive of the two measures, and in this application an undetected crack costs
more than a spurious one that the operator discards on inspection. The two weights
were fixed a priori and were never themselves optimised.

The search reported in the paper was given a budget of 300 evaluations and stopped
at evaluation 261, by which point the best score had improved by less than 0.01 over
the preceding seventy. It rose from 0.126, obtained with the hand-tuned starting
point, to 0.328. The winning parameter set is the one hard-coded in
``classical_pipeline.CFG``, so re-running this search is only necessary to reproduce
the search itself.

Note that these figures are not per-class segmentation scores and should not be
compared with the crater and crack metrics of the paper, which are computed on
separate channels by :mod:`tungsten_analyzer.metrics`.
"""

import argparse
import optuna
import cv2
import numpy as np
import os
import json
import concurrent.futures
from pathlib import Path
from tqdm import tqdm 

# The classical detector under optimisation.
from tungsten_analyzer import classical_pipeline as tsa

# ==========================================
# 1. EVALUATION HELPERS
# ==========================================
def json_a_mascara(json_path, shape_completa):
    """Rasterise a LabelMe annotation into a single boolean damage mask."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mask = np.zeros(shape_completa, dtype=np.uint8)
    for shape in data.get('shapes', []):
        puntos = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [puntos], 255)
    return mask > 127

def calcular_metricas(mascara_predicha, mascara_real):
    """Calcula IoU y F1-Score."""
    y_pred = mascara_predicha.astype(bool)
    y_true = mascara_real.astype(bool)
    interseccion = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    
    iou = interseccion / union if union > 0 else (1.0 if y_pred.sum() == 0 else 0.0)
    suma_areas = y_pred.sum() + y_true.sum()
    f1_score = (2.0 * interseccion) / suma_areas if suma_areas > 0 else (1.0 if y_pred.sum() == 0 else 0.0)
    
    nota_hibrida = (iou * 0.3) + (f1_score * 0.7)
    return nota_hibrida, iou, f1_score

# ==========================================
# 2. CARGA INTELIGENTE EN RAM
# ==========================================
def cargar_dataset(ruta_dataset):
    print("\n  Loading dataset into memory, resolving scales once for all trials...")
    dataset = []
    ruta_imgs = Path(ruta_dataset) / "originales"
    ruta_gts = Path(ruta_dataset) / "ground_truths"
    
    if not ruta_imgs.exists() or not ruta_gts.exists():
        print("  Error: the 'originales' and 'ground_truths' directories are missing.")
        return dataset

    archivos = list(ruta_imgs.glob("*.jpg"))
    for archivo in tqdm(archivos, desc="Cargando dataset", bar_format="{l_bar}{bar:20}{r_bar}"):
        gt_path = str(ruta_gts / (archivo.stem + ".json"))
        img_path = str(archivo)
        
        if os.path.exists(gt_path):
            img_bgr, img_gray = tsa.load_image(img_path)
            meta_h = tsa.detect_meta_strip(img_gray)
            
            # Running the OCR once per image and caching the result is what makes
            # the search affordable, since every trial revisits the same images.
            # If the OCR fails we fall back to 0.2236 um/px, the value for 500x.
            # (Revision 1: this fallback is kept only so that the search can be
            # reproduced as it was run; the analysis tools no longer use a default
            # pixel size, see measurement.read_scale.)
            um_px = tsa.auto_scale(img_gray, meta_h)
            if um_px is None:
                um_px = 0.2236 
                
            mascara_completa = json_a_mascara(gt_path, img_gray.shape)
            
            dataset.append({
                'nombre': archivo.name,
                'path': img_path,
                'img_bgr': img_bgr,
                'gt_mask': mascara_completa,
                'um_px': um_px  # Guardamos la escala
            })
            
    print(f"  Dataset ready: {len(dataset)} images pre-processed.")
    return dataset

# ==========================================
# 3. OBJECTIVE FUNCTION
# ==========================================
def objective(trial, dataset, barra_global):
    
    # Inject the mutated parameters into the classical pipeline configuration.
    tsa.CFG["ws_open_init"] = trial.suggest_int("ws_open_init", 3, 12)
    tsa.CFG["ws_min_sum"] = trial.suggest_int("ws_min_sum", 20, 100)
    tsa.CFG["ws_close"] = trial.suggest_int("ws_close", 5, 30)
    tsa.CFG["ws_open_sec"] = trial.suggest_int("ws_open_sec", 2, 8)
    tsa.CFG["ws_seed_pct"] = trial.suggest_int("ws_seed_pct", 20, 60)
    tsa.CFG["ws_firewall"] = trial.suggest_float("ws_firewall", 0.01, 0.08)
    tsa.CFG["dark_sens"] = trial.suggest_float("dark_sens", -0.70, -0.30)
    tsa.CFG["dark_open"] = trial.suggest_int("dark_open", 5, 40)
    tsa.CFG["dark_exp"] = trial.suggest_int("dark_exp", 40, 120)
    tsa.CFG["bright_sens"] = trial.suggest_float("bright_sens", 0.75, 1.2)
    tsa.CFG["bright_close"] = trial.suggest_int("bright_close", 2, 8)
    tsa.CFG["bright_open"] = trial.suggest_int("bright_open", 2, 10)
    tsa.CFG["bright_exp"] = trial.suggest_int("bright_exp", 40, 120)
    tsa.CFG["abl_blur"] = trial.suggest_int("abl_blur", 11, 31, step=2) 
    tsa.CFG["abl_sens"] = trial.suggest_float("abl_sens", 0.15, 0.50)
    tsa.CFG["abl_open"] = trial.suggest_int("abl_open", 5, 15)
    tsa.CFG["abl_exp"] = trial.suggest_int("abl_exp", 150, 400)
    tsa.CFG["rough_grad"] = trial.suggest_int("rough_grad", 2, 5)
    tsa.CFG["rough_std"] = trial.suggest_float("rough_std", 1.5, 3.5)
    tsa.CFG["rough_open"] = trial.suggest_int("rough_open", 5, 20)
    tsa.CFG["rough_exp"] = trial.suggest_int("rough_exp", 40, 120)
    tsa.CFG["min_crater_area"] = trial.suggest_int("min_crater_area", 200, 600)
    tsa.CFG["vapor_ring"] = trial.suggest_int("vapor_ring", 3, 8)
    tsa.CFG["vapor_circ"] = trial.suggest_float("vapor_circ", 0.20, 0.50)
    tsa.CFG["art_zone1"] = trial.suggest_float("art_zone1", 0.35, 0.55)
    tsa.CFG["art_zone2"] = trial.suggest_float("art_zone2", 0.15, 0.35)
    tsa.CFG["art_int"] = trial.suggest_int("art_int", 70, 110)
    tsa.CFG["frangi_start"] = trial.suggest_float("frangi_start", 0.3, 1.0)
    tsa.CFG["frangi_stop"] = trial.suggest_float("frangi_stop", 6.0, 12.0)
    tsa.CFG["bh_kernel"] = trial.suggest_int("bh_kernel", 9, 21, step=2) 
    tsa.CFG["crack_h_out"] = trial.suggest_float("crack_h_out", 0.05, 0.20)
    tsa.CFG["crack_l_out"] = trial.suggest_float("crack_l_out", 0.001, 0.05)
    tsa.CFG["crack_h_in"] = trial.suggest_float("crack_h_in", 0.15, 0.40)
    tsa.CFG["crack_l_in"] = trial.suggest_float("crack_l_in", 0.02, 0.10)
    tsa.CFG["bridge_h"] = trial.suggest_int("bridge_h", 3, 7)
    tsa.CFG["bridge_v"] = trial.suggest_int("bridge_v", 10, 20)
    tsa.CFG["min_crack_area"] = trial.suggest_int("min_crack_area", 15, 60)
    tsa.CFG["surv_v"] = trial.suggest_int("surv_v", 25, 45)
    tsa.CFG["surv_h"] = trial.suggest_int("surv_h", 45, 80)
    tsa.CFG["min_skel"] = trial.suggest_int("min_skel", 10, 25)

    notas_hibridas = []
    notas_iou = []
    notas_f1 = []
    
    # Track the best-scoring image of this particular trial, for the visual callback.
    mejor_nota_local = -1.0
    mejor_idx_local = 0
    
    barra_lote = tqdm(total=len(dataset), desc=f"Gen {trial.number}", leave=False, bar_format="{l_bar}{bar:20}{r_bar}")
    
    def evaluar_imagen(args):
        idx, data = args
        try:
            resultados = tsa.analyze(
                img_path=data['path'],
                um_per_px=data['um_px'], 
                verbose=False,
                output_dir=None
            )
            nota_hibrida, iou, f1 = calcular_metricas(resultados['mascara_total_booleana'], data['gt_mask'])
            return idx, nota_hibrida, iou, f1
        except Exception as e:
            return idx, None, None, None 

    with concurrent.futures.ThreadPoolExecutor(max_workers=22) as executor:
        # enumerate keeps each worker aware of which image it is processing.
        resultados_hilos = executor.map(evaluar_imagen, enumerate(dataset))
        
        for resultado in resultados_hilos:
            idx, nota_hibrida, iou, f1 = resultado
            if nota_hibrida is None:
                trial.set_user_attr("iou_puro", 0.0)
                trial.set_user_attr("f1_puro", 0.0)
                barra_lote.close()
                return 0.0
                
            notas_hibridas.append(nota_hibrida)
            notas_iou.append(iou)
            notas_f1.append(f1)
            
            # Remember the best image of this trial so the callback can render it.
            if nota_hibrida > mejor_nota_local:
                mejor_nota_local = nota_hibrida
                mejor_idx_local = idx
                
            barra_lote.update(1)
            
    barra_lote.close()
    barra_global.update(1)

    trial.set_user_attr("iou_puro", np.mean(notas_iou))
    trial.set_user_attr("f1_puro", np.mean(notas_f1))
    
    # Hand the callback the winning image of this trial together with its score.
    trial.set_user_attr("mejor_idx_img", mejor_idx_local)
    trial.set_user_attr("mejor_nota_img", mejor_nota_local)

    return np.mean(notas_hibridas)

# ==========================================
# 4. VISUAL CALLBACK
# ==========================================
def guardar_mejor_imagen(study, trial):
    # Only act when the global record has been beaten.
    if study.best_trial.number == trial.number:
        print("\n" + "="*50)
        print(f"  New best at trial {trial.number}")
        print(f"   Composite score: {trial.value:.4f}")
        print(f"   (IoU Puro: {trial.user_attrs['iou_puro']:.4f} | F1-Score Puro: {trial.user_attrs['f1_puro']:.4f})")
        print("="*50 + "\n")
        
        try:
            # Recover the image that segmented best in this trial.
            mejor_idx = trial.user_attrs.get("mejor_idx_img", 0)
            mejor_nota = trial.user_attrs.get("mejor_nota_img", 0.0)
            muestra = dataset[mejor_idx] # Cogemos esa foto concreta
            
            # Re-run the pipeline on it to obtain the masks for the figure.
            res = tsa.analyze(img_path=muestra['path'], um_per_px=muestra['um_px'], verbose=False, output_dir=None)
            
            mask_real = muestra['gt_mask']
            img_bgr = muestra['img_bgr'].copy()
            h, w = img_bgr.shape[:2]
            
            # Creamos un lienzo doble
            lienzo = np.zeros((h, w * 2, 3), dtype=np.uint8)
            
            # ----------------------------------------------------
            # IMAGEN IZQUIERDA: Ground Truth
            # ----------------------------------------------------
            img_gt = img_bgr.copy()
            overlay_gt = np.zeros_like(img_bgr)
            overlay_gt[mask_real] = [0, 255, 0] # BGR: Verde
            cv2.addWeighted(overlay_gt, 0.4, img_gt, 1.0, 0, img_gt)
            cv2.putText(img_gt, f"GROUND TRUTH - {muestra['nombre']}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # ----------------------------------------------------
            # Right panel: prediction, cracks in red and craters in green.
            # ----------------------------------------------------
            img_pred = img_bgr.copy()
            overlay_pred = np.zeros_like(img_bgr)
            
            # Look for the separate masks in the dictionary returned by analyze().
            tiene_separadas = False
            
            # Adjust these keys if analyze() renames its mask entries.
            if 'mascara_crateres' in res and 'mascara_grietas' in res:
                # OJO con OpenCV: Los colores son BGR (Blue, Green, Red)
                overlay_pred[res['mascara_crateres'].astype(bool)] = [0, 255, 0] # Verde
                overlay_pred[res['mascara_grietas'].astype(bool)] = [0, 0, 255]  # Rojo
                tiene_separadas = True
            elif 'mascara_crateres_booleana' in res and 'mascara_grietas_booleana' in res:
                overlay_pred[res['mascara_crateres_booleana']] = [0, 255, 0] 
                overlay_pred[res['mascara_grietas_booleana']] = [0, 0, 255]  
                tiene_separadas = True
            else:
                # Fall back to the merged mask when the separate ones are absent.
                overlay_pred[res['mascara_total_booleana']] = [0, 0, 255] 
            
            cv2.addWeighted(overlay_pred, 0.4, img_pred, 1.0, 0, img_pred)
            cv2.putText(img_pred, f"PRED Gen {trial.number} | Nota Foto: {mejor_nota:.3f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            if not tiene_separadas:
                cv2.putText(img_pred, "(analyze() did not return separate masks)", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Unimos y guardamos
            lienzo[:, :w] = img_gt
            lienzo[:, w:] = img_pred
            
            nombre_salida = f"RECORD_Gen_{trial.number}_IoU_{trial.user_attrs['iou_puro']:.3f}.png"
            ruta_salida = Path("./Evolucion_Visual") / nombre_salida
            cv2.imwrite(str(ruta_salida), lienzo)
            print(f"   Snapshot saved for {muestra['nombre']}, score {mejor_nota:.3f}")
            
        except Exception as e:
            print(f"  Could not render the comparison figure: {e}")

# ==========================================
# 5. MOTOR PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # Silence Optuna's own logging so it does not fight with the progress bar.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True,
                        help="directory holding 'originales' and 'ground_truths'")
    parser.add_argument("--trials", type=int, default=300,
                        help="evaluation budget (the paper stopped at 261)")
    parser.add_argument("--out", default="./search_snapshots",
                        help="where to write the figure of each new best individual")
    args = parser.parse_args()

    RUTA_DATASET = args.dataset
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    # The dataset is global so that the callback can reach the images.
    global dataset
    dataset = cargar_dataset(RUTA_DATASET)
    
    if len(dataset) > 0:
        NUM_GENERACIONES = args.trials
        
        sampler_genetico = optuna.samplers.NSGAIISampler(population_size=20, mutation_prob=0.1)
        study = optuna.create_study(direction="maximize", sampler=sampler_genetico, study_name="Evolucion_Parametros_SEM")
        
        print(f"\n  Starting evolutionary search (Optuna NSGA-II), budget {NUM_GENERACIONES} evaluations")
        
        with tqdm(total=NUM_GENERACIONES, desc="Search", position=0, bar_format="{l_bar}{bar:30}{r_bar}", colour="green") as pbar_global:
            study.optimize(
                lambda trial: objective(trial, dataset, pbar_global),
                n_trials=NUM_GENERACIONES,
                show_progress_bar=False, 
                n_jobs=1, # 
                callbacks=[guardar_mejor_imagen]
            )
        
        print("\n" + "="*50)
        print("SEARCH COMPLETE")
        print(f"Best composite score: {study.best_value:.4f}")
        try:
            print(f"IoU Puro: {study.best_trial.user_attrs['iou_puro']:.4f}")
            print(f"F1-Score Puro: {study.best_trial.user_attrs['f1_puro']:.4f}")
        except:
            pass
        
        print("\nBest parameter set:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print("="*50)