import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

def comprobar_hardware():
    print("\n--- 🔍 PIPELINE DE HARDWARE EN PRODUCCIÓN ---")
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if dispositivo == "cuda":
        print(f"🎮 GPU Activa: {torch.cuda.get_device_name(0)}")
        print("⚡ Modo de entrenamiento masivo: ACTIVADO")
    else:
        print("⚠️ ALERTA: No se detectó GPU CUDA. El proceso será extremadamente lento.")
    print("---------------------------------------------\n")
    return dispositivo

def iniciar_entrenamiento_masivo():
    dispositivo = comprobar_hardware()
    
    # 1. Cargar el Dataset REAL COMPLETO (Sin recortes)
    print("📥 Descargando/Cargando dataset 'ag_news' completo desde el Hub...")
    dataset_train = load_dataset("ag_news", split="train") # 120,000 filas
    dataset_val = load_dataset("ag_news", split="test")   # 7,600 filas
    
    num_clases = 4 # (0: World, 1: Sports, 2: Business, 3: Sci/Tech)
    
    # 2. Tokenizador y Arquitectura Eficiente
    print("🧠 Cargando arquitectura DistilBERT y Tokenizador...")
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    modelo = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_clases)
    
    # Reducimos max_length a 96 para acelerar el entrenamiento en tu GPU sin perder información
    def tokenizar_datos(ejemplos):
        return tokenizer(ejemplos["text"], truncation=True, padding="max_length", max_length=96)
    
    print("🪙 Tokenizando las 120,000 filas en bloques optimizados...")
    train_tokenizado = dataset_train.map(tokenizar_datos, batched=True)
    val_tokenizado = dataset_val.map(tokenizar_datos, batched=True)
    
    # 3. Métrica de Evaluación Profesional
    metrica = evaluate.load("accuracy")
    def calcular_metricas(eval_pred):
        logits, labels = eval_pred
        predicciones = np.argmax(logits, axis=-1)
        return metrica.compute(predictions=predicciones, references=labels)
    
    # 4. Hiperparámetros de Big Tech Optimizados para GTX 1650 (4GB VRAM)
    print("⚙️ Configurando argumentos de entrenamiento en lote continuo...")
    ruta_guardado = "./models/clasificador_noticias_produccion"
    
    argumentos_entrenamiento = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",             # Evaluar cada ciertos pasos para monitorear progreso largo
        eval_steps=500,                         # Evalúa cada 500 pasos
        save_strategy="steps",
        save_steps=500,                         # Guarda un checkpoint cada 500 pasos por seguridad
        learning_rate=3e-5,                     # Ajustado para convergencia estable
        per_device_train_batch_size=4,          # Cuidado estricto de tus 4GB de VRAM
        per_device_eval_batch_size=4,
        num_train_epochs=1,                     # 1 época completa es suficiente para 120,000 filas
        weight_decay=0.01,
        fp16=True if dispositivo == "cuda" else False, # Precisión mixta para exprimir tus núcleos CUDA
        logging_steps=100,                      # Te muestra cómo va la pérdida en la terminal cada 100 pasos
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,                     # Solo mantiene los 2 mejores checkpoints para no llenar tu disco
        report_to="none"
    )
    
    # 5. Inicializar el Trainer de Hugging Face
    trainer = Trainer(
        model=modelo,
        args=argumentos_entrenamiento,
        train_dataset=train_tokenizado,
        eval_dataset=val_tokenizado,
        compute_metrics=calcular_metricas,
    )
    
    # 6. ¡Lanzar el entrenamiento pesado! 🔥
    print("\n🚀 ¡Iniciando el entrenamiento real de 120,000 filas en tu GPU!")
    print("⏱️ Esto tomará alrededor de 20-30 minutos. Monitorea los logs a continuación:\n")
    trainer.train()
    
    # 7. Guardar el Modelo Final de Producción
    print(f"\n💾 Guardando el artefacto final optimizado en: {ruta_guardado}")
    modelo.save_pretrained(ruta_guardado)
    tokenizer.save_pretrained(ruta_guardado)
    print("🎉 ¡Hito alcanzado! El modelo real está listo para producción.")

if __name__ == "__main__":
    iniciar_entrenamiento_masivo()