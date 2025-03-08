# Este script actuará como un wrapper para SnakeNeuralSystem.py
# Se enfoca en inicializar correctamente y mostrar mensajes de diagnóstico

import sys
import time
import traceback

print("Iniciando script de depuración para SnakeNeuralSystem...")
print(f"Python version: {sys.version}")

try:
    # Intentar importar pygame primero para verificar instalación
    print("Verificando instalación de pygame...")
    import pygame
    print(f"Pygame version: {pygame.version.ver}")
    pygame.init()
    print("Pygame inicializado correctamente")
    
    # Verificar numpy
    print("Verificando instalación de numpy...")
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # Verificar torch
    print("Verificando instalación de PyTorch...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Verificar que SnakeRL.py existe y la clase SnakeGame es accesible
    print("Intentando importar SnakeGame desde SnakeRL.py...")
    try:
        from SnakeRL import SnakeGame
        print("SnakeGame imported successfully!")
        
        # Verificar que se puede crear una instancia
        test_env = SnakeGame(render=True, difficulty=0)
        print("¡SnakeGame inicializado correctamente!")
        
        # Limpiar antes de continuar
        pygame.quit()
        
    except ImportError as e:
        print(f"ERROR: No se pudo importar SnakeGame: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR al inicializar SnakeGame: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Ahora importa el módulo principal
    print("\nImportando SnakeNeuralAdvanced.py...")
    try:
        import SnakeNeuralAdvanced
        print("Módulo SnakeNeuralAdvanced importado correctamente!")
        
        # Intentar ejecutar la función principal con un mecanismo de seguridad
        print("\nEjecutando integrate_with_snakeRL con un tiempo máximo de ejecución...")
        
        # Este enfoque es simple, solo ejecuta la función
        # No establecemos un timeout ya que queremos ver dónde se atasca
        SnakeNeuralAdvanced.integrate_with_snakeRL()
        
    except Exception as e:
        print(f"\nERROR durante la ejecución: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"ERROR en inicialización básica: {e}")
    traceback.print_exc()
finally:
    print("\nScript de depuración finalizado.")
    time.sleep(5)  # Mantener la consola abierta por 5 segundos