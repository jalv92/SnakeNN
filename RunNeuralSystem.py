# Script para ejecutar el sistema neural avanzado con el SnakeRL recreado
import os
import sys
import time
import traceback
import argparse

# Crear un parser de argumentos para permitir opciones por línea de comandos
parser = argparse.ArgumentParser(description='Ejecutar el sistema neural avanzado para Snake')
parser.add_argument('--model', type=str, default='snake_agent_fixed.pth', help='Ruta al modelo a cargar')
parser.add_argument('--episodes', type=int, default=1000, help='Número de episodios para entrenar')
parser.add_argument('--difficulty', type=int, default=0, choices=[0, 1, 2], 
                   help='Dificultad del entorno (0: sin laberinto, 1: laberinto simple, 2: laberinto complejo)')
parser.add_argument('--new-agent', action='store_true', help='Crear un nuevo agente en lugar de cargar uno existente')

args = parser.parse_args()

print(f"\n{'='*60}")
print(f"SISTEMA NEURAL AVANZADO PARA SNAKE")
print(f"{'='*60}")

# Verificar requisitos necesarios
print("\nVerificando requisitos...")
try:
    import pygame
    print(f"✓ Pygame {pygame.version.ver}")
    
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
    
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA: {'✓ disponible' if torch.cuda.is_available() else '✗ no disponible'}")
except ImportError as e:
    print(f"ERROR: Falta alguna biblioteca necesaria - {e}")
    print("Instala las bibliotecas faltantes e intenta de nuevo.")
    sys.exit(1)

# Verificar que el archivo SnakeRL.py está disponible
print("\nVerificando SnakeRL.py...")
if not os.path.exists("SnakeRL.py"):
    print("ERROR: No se encuentra el archivo SnakeRL.py")
    print("Asegúrate de que el archivo recreado SnakeRL.py esté en el mismo directorio.")
    sys.exit(1)

# Verificar que el archivo SnakeNeuralAdvanced.py está disponible
print("Verificando SnakeNeuralAdvanced.py...")
if not os.path.exists("SnakeNeuralAdvanced.py"):
    print("ERROR: No se encuentra el archivo SnakeNeuralAdvanced.py")
    print("Crea este archivo con el sistema neural avanzado.")
    sys.exit(1)

# Verificar que el modelo existe
if not args.new_agent and not os.path.exists(args.model):
    print(f"ERROR: No se encuentra el archivo del modelo {args.model}")
    print("Asegúrate de que el archivo del modelo exista o usa --new-agent para crear uno nuevo.")
    sys.exit(1)

# Verificar que se puede importar SnakeGame
try:
    from SnakeRL import SnakeGame
    print("✓ SnakeGame importado correctamente")
    
    # Probar inicialización
    test_env = SnakeGame(render=False, difficulty=0)
    test_env.reset()
    print("✓ SnakeGame inicializado correctamente")
    
except ImportError as e:
    print(f"ERROR: No se pudo importar SnakeGame - {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR al inicializar SnakeGame: {e}")
    traceback.print_exc()
    sys.exit(1)

# Todo listo, ahora importar y ejecutar el sistema neural
print("\nTodo listo para ejecutar el sistema neural avanzado...")
print("Iniciando en 3 segundos...")
time.sleep(3)

try:
    # Importar el módulo del sistema neural
    import SnakeNeuralAdvanced
    
    # Ejecutar la función principal
    print("\nEjecutando Sistema Neural Avanzado:")
    if hasattr(SnakeNeuralAdvanced, 'integrate_with_snakeRL'):
        SnakeNeuralAdvanced.integrate_with_snakeRL(
            episodes=args.episodes,
            save_path=args.model,
            load_existing=not args.new_agent,
            difficulty=args.difficulty
        )
    else:
        print("ERROR: No se encontró la función 'integrate_with_snakeRL' en el módulo.")
        print("Verifica que el archivo SnakeNeuralAdvanced.py contenga esta función.")
        
except Exception as e:
    print(f"\nERROR durante la ejecución: {e}")
    traceback.print_exc()
finally:
    # Limpiar al salir
    try:
        pygame.quit()
    except:
        pass
    
    print("\nFin de la ejecución")