# Script para probar la implementación recreada de SnakeGame
import sys
import time
import random
import pygame

try:
    # Importar la clase SnakeGame desde el archivo recreado
    from SnakeRL import SnakeGame, UP, DOWN, LEFT, RIGHT
    print("SnakeGame importado correctamente")
except ImportError as e:
    print(f"Error al importar SnakeGame: {e}")
    print("Asegúrate de que el archivo SnakeRL.py está en el mismo directorio")
    sys.exit(1)

def run_manual_snake():
    """Ejecuta el juego Snake permitiendo control manual con las flechas del teclado"""
    # Inicializar juego
    env = SnakeGame(render=True, difficulty=1)
    state = env.reset()
    
    print("Juego Snake inicializado. Controles:")
    print("- Flechas: Mover serpiente")
    print("- ESC: Salir")
    print("- M: Aumentar velocidad")
    print("- N: Reducir velocidad")
    
    done = False
    score = 0
    
    # Mapeo de teclas a acciones
    action_map = {}  # Se llenará durante el juego
    
    while not done:
        # Por defecto seguir recto
        action = 0
        
        # Procesar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                # Determinar la acción basada en la dirección actual y la tecla presionada
                elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    current_direction = env.direction
                    
                    # Calcular el mapeo de teclas a acciones según la dirección actual
                    if current_direction == UP:
                        action_map = {
                            pygame.K_UP: 0,      # Seguir recto (arriba)
                            pygame.K_RIGHT: 2,   # Girar a la derecha
                            pygame.K_LEFT: 1,    # Girar a la izquierda
                            pygame.K_DOWN: 0     # No permitir dar la vuelta completa
                        }
                    elif current_direction == RIGHT:
                        action_map = {
                            pygame.K_RIGHT: 0,   # Seguir recto (derecha)
                            pygame.K_DOWN: 2,    # Girar a la derecha
                            pygame.K_UP: 1,      # Girar a la izquierda
                            pygame.K_LEFT: 0     # No permitir dar la vuelta completa
                        }
                    elif current_direction == DOWN:
                        action_map = {
                            pygame.K_DOWN: 0,    # Seguir recto (abajo)
                            pygame.K_LEFT: 2,    # Girar a la derecha
                            pygame.K_RIGHT: 1,   # Girar a la izquierda
                            pygame.K_UP: 0       # No permitir dar la vuelta completa
                        }
                    elif current_direction == LEFT:
                        action_map = {
                            pygame.K_LEFT: 0,    # Seguir recto (izquierda)
                            pygame.K_UP: 2,      # Girar a la derecha
                            pygame.K_DOWN: 1,    # Girar a la izquierda
                            pygame.K_RIGHT: 0    # No permitir dar la vuelta completa
                        }
                    
                    # Aplicar la acción correspondiente
                    action = action_map.get(event.key, 0)
        
        # Ejecutar acción
        next_state, reward, done = env.step(action)
        
        # Actualizar estado
        state = next_state
        
        # Renderizar el juego
        env.render(env.score, score, 0)
        
        # Si el juego termina, reiniciar después de una pausa
        if done:
            print(f"Juego terminado! Puntuación: {env.score}")
            time.sleep(1)
            state = env.reset()
            done = False
        
    pygame.quit()
    print("Juego finalizado")

def run_random_agent():
    """Ejecuta el juego Snake con un agente aleatorio"""
    env = SnakeGame(render=True, difficulty=1)
    state = env.reset()
    
    print("Ejecutando agente aleatorio. Presiona ESC para salir.")
    
    total_score = 0
    episodes = 10
    
    for episode in range(episodes):
        done = False
        episode_steps = 0
        
        while not done:
            # Acción aleatoria: 0 (recto), 1 (izquierda) o 2 (derecha)
            action = random.randint(0, 2)
            
            # Ejecutar acción
            next_state, reward, done = env.step(action)
            
            # Actualizar estado
            state = next_state
            episode_steps += 1
            
            # Renderizar el juego
            env.render(env.score, total_score, episodes - episode - 1)
            
            # Manejar eventos
            if not env.handle_events():
                pygame.quit()
                return
        
        total_score += env.score
        print(f"Episodio {episode+1}/{episodes} - Score: {env.score} - Pasos: {episode_steps}")
        
        # Reiniciar para el siguiente episodio
        if episode < episodes - 1:
            state = env.reset()
    
    pygame.quit()
    print(f"Agente aleatorio completado: {episodes} episodios, score promedio: {total_score/episodes:.2f}")

if __name__ == "__main__":
    print("¿Qué modo deseas ejecutar?")
    print("1. Modo manual (controlar con las flechas)")
    print("2. Agente aleatorio")
    
    choice = input("Selecciona una opción (1-2): ")
    
    if choice == "1":
        run_manual_snake()
    elif choice == "2":
        run_random_agent()
    else:
        print("Opción no válida")