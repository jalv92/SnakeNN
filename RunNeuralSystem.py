# Script para ejecutar el sistema neural avanzado con el SnakeRL recreado
import os
import sys
import time
import traceback

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
        SnakeNeuralAdvanced.integrate_with_snakeRL()
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