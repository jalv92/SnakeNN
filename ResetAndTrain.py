import os
import sys
import time
import argparse
import torch

from SnakeNeuralAdvanced import (
    create_advanced_agent, 
    integrate_with_snakeRL,
    save_advanced_agent
)

def reset_and_train(episodes=1000, save_path="snake_agent_new.pth", difficulty=0):
    """
    Reinicia completamente el entrenamiento con un agente nuevo.
    
    Args:
        episodes: Número de episodios para entrenar
        save_path: Ruta donde guardar el agente entrenado
        difficulty: Nivel de dificultad del entorno (0-2)
    """
    print(f"\n{'='*60}")
    print(f"REINICIO Y ENTRENAMIENTO DE AGENTE NEURAL")
    print(f"{'='*60}\n")
    
    # Limpiar la memoria CUDA antes de empezar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Memoria CUDA limpiada")
    
    # Integrar con SnakeRL pero forzando un agente nuevo
    print(f"Iniciando entrenamiento con un agente NUEVO por {episodes} episodios")
    print(f"Nivel de dificultad: {difficulty}")
    print(f"El modelo se guardará en: {save_path}\n")
    
    # Pequeña pausa para que el usuario pueda leer la información
    time.sleep(2)
    
    # Iniciar entrenamiento con un agente nuevo
    integrate_with_snakeRL(
        episodes=episodes,
        save_path=save_path,
        load_existing=False,  # Forzar agente nuevo
        difficulty=difficulty
    )
    
    print("\nEntrenamiento completado.")
    print(f"Agente guardado en: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reiniciar entrenamiento con un agente nuevo")
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episodios para entrenar")
    parser.add_argument("--save-path", type=str, default="snake_agent_new.pth", help="Ruta donde guardar el agente")
    parser.add_argument("--difficulty", type=int, default=0, choices=[0, 1, 2], 
                       help="Dificultad (0: sin laberinto, 1: laberinto simple, 2: laberinto complejo)")
    
    args = parser.parse_args()
    
    reset_and_train(
        episodes=args.episodes,
        save_path=args.save_path,
        difficulty=args.difficulty
    ) 