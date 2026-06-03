import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def comprobar_gpu():
    print("\n--- 🔍 VERIFICACIÓN DE HARDWARE ---")
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo detectado para entrenamiento: {dispositivo.upper()}")
    if dispositivo == "cuda":
        print(f"🎮 Entrenando en: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ ALERTA: No se detectó GPU. El entrenamiento en CPU será lento.")
    print("-----------------------------------\n")
    return dispositivo

def preparar_datos_ejemplo():
    # Dataset de juguete ultra-ligero para validar que todo funcione
    datos = {
        "text": [
            "El modelo de machine learning funciona espectacular en mi GPU",
            "Tengo un error en el pipeline de CI/CD y no compila",
            "La red neuronal tardó tres horas en entrenar en la laptop",
            "El servidor de producción se cayó por falta de memoria"
        ],
        "label": [0, 1, 0, 1]  # 0: Éxito/ML, 1: Error/DevOps
    }
    return Dataset.from_dict(datos)

def iniciar_entrenamiento():
    dispositivo = comprobar_gpu()
    
    # 1. Cargar Tokenizer y Modelo Base ligero (DistilBERT)
    print("📥 Cargando arquitectura DistilBERT...")
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    modelo = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    # 2. Preparar los datos
    dataset = preparar_datos_ejemplo()
    
    def tokenizar_funcion(ejemplos):
        return tokenizer(ejemplos["text"], truncation=True, padding="max_length", max_length=64)
    
    print("🪙 Tokenizando textos de prueba...")
    dataset_tokenizado = dataset.map(tokenizar_funcion, batched=True)
    
    # 3. Configurar Argumentos de Entrenamiento optimizados para tu GTX 1650
    print("⚙️ Configurando hiperparámetros para la GPU...")
    argumentos_entrenamiento = TrainingArguments(
        output_dir="./results",                  # Carpeta temporal de resultados
        num_train_epochs=3,                     # 3 vueltas completas al dataset
        per_device_train_batch_size=2,          # Batch pequeño para cuidar tus 4GB de VRAM
        fp16=True if dispositivo == "cuda" else False, # ¡Precisión mixta activada para ir volando!
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="no",
        report_to="none"                        # Evita registrar en plataformas externas por ahora
    )
    
    # 4. Inicializar el Trainer de Hugging Face
    trainer = Trainer(
        model=modelo,
        args=argumentos_entrenamiento,
        train_dataset=dataset_tokenizado,
    )
    
    # 5. ¡FUEGO! 🔥
    print("\n🚀 ¡Lanzando bucle de entrenamiento en los núcleos CUDA!")
    trainer.train()
    print("\n🎉 ¡Entrenamiento completado con éxito absoluto!")

if __name__ == "__main__":
    iniciar_entrenamiento()