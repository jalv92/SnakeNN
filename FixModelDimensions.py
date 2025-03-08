import os
import torch
import argparse
import numpy as np
import time
import gc

# Importar nuestro módulo de IA
from SnakeNeuralAdvanced import (
    create_advanced_agent, 
    load_advanced_agent, 
    save_advanced_agent,
    PerceptionModule,
    NavigationModule,
    ExecutiveModule,
    PredictionModule,
    ModularAdvancedAgent,
    DynamicLayer,
    integrate_with_snakeRL
)

def fix_model_dimensions(model_path, output_path=None):
    """
    Crea un nuevo agente con las dimensiones correctas basadas en el modelo guardado.
    
    Args:
        model_path: Ruta al modelo guardado
        output_path: Ruta para guardar el modelo corregido (opcional)
    
    Returns:
        True si la corrección fue exitosa, False en caso contrario
    """
    if not os.path.exists(model_path):
        print(f"Error: El archivo del modelo {model_path} no existe.")
        return False
    
    if output_path is None:
        # Usar el mismo nombre con un sufijo
        output_path = model_path.replace('.pth', '_fixed.pth')
    
    print(f"Analizando modelo: {model_path}")
    
    try:
        # Cargar el modelo original para análisis
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Extraer las dimensiones de las capas principales
        module_dimensions = {}
        
        # Analizar el módulo de percepción
        if 'perception' in state_dict['modules']:
            perception_state = state_dict['modules']['perception']
            module_dimensions['perception'] = {
                'input_size': perception_state['feature_extractor.0.weight'].shape[1],
                'hidden_size': perception_state['feature_extractor.0.weight'].shape[0],
                'output_size': perception_state['feature_extractor.2.weight'].shape[0]
            }
        
        # Analizar el módulo de navegación
        if 'navigation' in state_dict['modules']:
            navigation_state = state_dict['modules']['navigation']
            module_dimensions['navigation'] = {
                'input_size': navigation_state['pathfinder.0.weight'].shape[1],
                'hidden_size': navigation_state['pathfinder.2.weight'].shape[0],
                'output_size': navigation_state['pathfinder.4.weight'].shape[0]
            }
        
        # Analizar el módulo de ejecutivo
        if 'executive' in state_dict['modules']:
            executive_state = state_dict['modules']['executive']
            module_dimensions['executive'] = {
                'input_size': executive_state['integrator.0.weight'].shape[1],
                'hidden_size': executive_state['integrator.2.weight'].shape[0],
                'output_size': executive_state['action_selector.weight'].shape[0]
            }
            
        # Analizar el módulo de predicción si existe
        if 'prediction' in state_dict['modules']:
            prediction_state = state_dict['modules']['prediction']
            if 'state_predictor.0.weight' in prediction_state:
                module_dimensions['prediction'] = {
                    'input_size': prediction_state['state_predictor.0.weight'].shape[1],
                    'hidden_size': prediction_state['state_predictor.0.weight'].shape[0],
                    'output_size': prediction_state['state_predictor.2.weight'].shape[0]
                }
        
        # Dimensiones básicas
        state_size = module_dimensions['perception']['input_size'] if 'perception' in module_dimensions else 38
        action_size = module_dimensions['executive']['output_size'] if 'executive' in module_dimensions else 3
        
        print("\nDimensiones detectadas:")
        for module, dims in module_dimensions.items():
            print(f"  {module.capitalize()}:")
            for key, value in dims.items():
                print(f"    {key}: {value}")
        
        # Verificar y corregir inconsistencias entre módulos conectados
        print("\nVerificando consistencia entre módulos conectados...")
        
        # Verificar la conexión entre percepción y navegación
        if 'perception' in module_dimensions and 'navigation' in module_dimensions:
            perception_output = module_dimensions['perception']['output_size']
            navigation_input_size = module_dimensions['navigation']['input_size']
            
            # El input de navegación debe incluir percepción + posición + objetivo (8)
            expected_nav_input = perception_output + 8  # Posición (2) + Objetivo (2) + Dirección (4)
            
            if navigation_input_size != expected_nav_input:
                print(f"  Inconsistencia detectada: Navegación espera {navigation_input_size} como entrada, pero debería ser {expected_nav_input}")
                module_dimensions['navigation']['input_size'] = expected_nav_input
                print(f"  Corrigiendo: Navegación input_size = {expected_nav_input}")
        
        # Verificar la conexión entre percepción, navegación y ejecutivo
        if 'perception' in module_dimensions and 'navigation' in module_dimensions and 'executive' in module_dimensions:
            perception_output = module_dimensions['perception']['output_size']
            navigation_output = module_dimensions['navigation']['output_size']
            prediction_output = module_dimensions['prediction']['output_size'] if 'prediction' in module_dimensions else 2
            
            # El input del ejecutivo debe ser la suma de las salidas de percepción, navegación y predicción
            expected_exec_input = perception_output + navigation_output + prediction_output
            
            executive_input = module_dimensions['executive']['input_size']
            
            if executive_input != expected_exec_input:
                print(f"  Inconsistencia detectada: Ejecutivo espera {executive_input} como entrada, pero debería ser {expected_exec_input}")
                module_dimensions['executive']['input_size'] = expected_exec_input
                print(f"  Corrigiendo: Ejecutivo input_size = {expected_exec_input}")
        
        print(f"\nCreando un nuevo agente con state_size={state_size}, action_size={action_size}")
        
        # Crear un agente vacío con las dimensiones básicas
        agent = ModularAdvancedAgent(state_size, action_size)
        
        # Ahora personalizar los módulos con las dimensiones correctas
        if 'perception' in module_dimensions:
            dims = module_dimensions['perception']
            # Reemplazar el módulo de percepción
            agent.modules['perception'] = PerceptionModule(
                input_size=dims['input_size'],
                output_size=dims['output_size']
            ).to(agent.device)
            
            # Ajustar la primera capa
            layer = agent.modules['perception'].feature_extractor[0]
            if isinstance(layer, DynamicLayer) and layer.current_output_size != dims['hidden_size']:
                print(f"  Ajustando primera capa de percepción: {layer.current_output_size} -> {dims['hidden_size']}")
                
                # Crear una nueva capa con dimensiones correctas
                new_layer = DynamicLayer(dims['input_size'], dims['hidden_size']).to(agent.device)
                agent.modules['perception'].feature_extractor[0] = new_layer
            
            # Ajustar la segunda capa
            layer = agent.modules['perception'].feature_extractor[2]
            if isinstance(layer, DynamicLayer) and layer.current_output_size != dims['output_size']:
                print(f"  Ajustando segunda capa de percepción: {layer.current_output_size} -> {dims['output_size']}")
                
                # Crear una nueva capa con dimensiones correctas
                new_layer = DynamicLayer(dims['hidden_size'], dims['output_size']).to(agent.device)
                agent.modules['perception'].feature_extractor[2] = new_layer
            
            # Actualizar el tamaño de salida
            agent.modules['perception'].output_size = dims['output_size']
            
            # Recrear attention con el tamaño correcto
            agent.modules['perception'].attention = torch.nn.Parameter(
                torch.ones(dims['output_size'], device=agent.device) / dims['output_size']
            )
        
        # Ajustar el módulo de navegación
        if 'navigation' in module_dimensions:
            dims = module_dimensions['navigation']
            
            # Recrear el módulo con las dimensiones correctas
            agent.modules['navigation'] = NavigationModule(
                input_size=dims['input_size'],
                output_size=dims['output_size']
            ).to(agent.device)
            
            # Ajustar capas intermedias
            pathfinder = agent.modules['navigation'].pathfinder
            
            # Primera capa
            if isinstance(pathfinder[0], DynamicLayer) and pathfinder[0].current_output_size != dims['hidden_size'] // 2:
                print(f"  Ajustando primera capa de navegación a tamaño {dims['hidden_size'] // 2}")
                pathfinder[0] = DynamicLayer(dims['input_size'], dims['hidden_size'] // 2).to(agent.device)
            
            # Segunda capa
            if isinstance(pathfinder[2], DynamicLayer) and pathfinder[2].current_output_size != dims['hidden_size']:
                print(f"  Ajustando segunda capa de navegación a tamaño {dims['hidden_size']}")
                pathfinder[2] = DynamicLayer(dims['hidden_size'] // 2, dims['hidden_size']).to(agent.device)
            
            # Tercera capa
            if isinstance(pathfinder[4], DynamicLayer) and pathfinder[4].current_output_size != dims['output_size']:
                print(f"  Ajustando tercera capa de navegación a tamaño {dims['output_size']}")
                pathfinder[4] = DynamicLayer(dims['hidden_size'], dims['output_size']).to(agent.device)
        
        # Ajustar el módulo ejecutivo
        if 'executive' in module_dimensions:
            dims = module_dimensions['executive']
            
            # Obtener los tamaños de los módulos conectados
            perception_size = module_dimensions['perception']['output_size'] if 'perception' in module_dimensions else 64
            navigation_size = module_dimensions['navigation']['output_size'] if 'navigation' in module_dimensions else 70
            prediction_size = module_dimensions['prediction']['output_size'] if 'prediction' in module_dimensions else 2
            
            # Recrear el módulo con las dimensiones correctas
            agent.modules['executive'] = ExecutiveModule(
                perception_size=perception_size,
                navigation_size=navigation_size,
                prediction_size=prediction_size,
                action_size=dims['output_size']
            ).to(agent.device)
            
            # Ajustar capas del integrador
            integrator = agent.modules['executive'].integrator
            
            # Primera capa del integrador
            if isinstance(integrator[0], DynamicLayer):
                input_size = perception_size + navigation_size + prediction_size
                print(f"  Ajustando primera capa del ejecutivo a tamaño de entrada {input_size}")
                integrator[0] = DynamicLayer(input_size, 64).to(agent.device)
            
            # Segunda capa del integrador
            if isinstance(integrator[2], DynamicLayer):
                print(f"  Ajustando segunda capa del ejecutivo a tamaño {dims['hidden_size']}")
                integrator[2] = DynamicLayer(64, dims['hidden_size']).to(agent.device)
            
            # Selector de acciones
            print(f"  Ajustando selector de acciones a entrada {dims['hidden_size']}, salida {dims['output_size']}")
            agent.modules['executive'].action_selector = DynamicLayer(
                dims['hidden_size'], dims['output_size']
            ).to(agent.device)
        
        # Ajustar el módulo de predicción si existe
        if 'prediction' in module_dimensions:
            dims = module_dimensions['prediction']
            
            # Verificar si existe un módulo de predicción ya
            if 'prediction' not in agent.modules:
                print("  Creando módulo de predicción")
                agent.modules['prediction'] = PredictionModule(
                    state_size=state_size,
                    action_size=action_size
                ).to(agent.device)
            
            # Ajustar dimensiones si es necesario
            prediction_module = agent.modules['prediction']
            
            if hasattr(prediction_module, 'state_predictor') and len(prediction_module.state_predictor) > 0:
                # Ajustar primera capa
                if isinstance(prediction_module.state_predictor[0], DynamicLayer):
                    print(f"  Ajustando primera capa de predicción a tamaño {dims['hidden_size']}")
                    prediction_module.state_predictor[0] = DynamicLayer(
                        dims['input_size'], dims['hidden_size']
                    ).to(agent.device)
                
                # Ajustar última capa
                if isinstance(prediction_module.state_predictor[2], DynamicLayer):
                    print(f"  Ajustando última capa de predicción a tamaño {dims['output_size']}")
                    prediction_module.state_predictor[2] = DynamicLayer(
                        dims['hidden_size'], dims['output_size']
                    ).to(agent.device)
        
        # Configurar los callbacks para propagación de cambios dimensionales
        print("\nConfigurando callbacks para propagación de cambios dimensionales...")
        agent._setup_size_change_listeners()
        
        # Intentar cargar el estado desde el modelo original
        print("\nIntentando cargar el estado desde el modelo original...")
        try:
            agent.load(model_path)
            print("Estado cargado correctamente.")
        except Exception as e:
            print(f"Error al cargar el estado: {e}")
            print("Se continuará con el agente redimensionado pero sin los pesos del modelo original.")
        
        # Guardar el modelo corregido
        print(f"\nGuardando modelo corregido en {output_path}")
        save_advanced_agent(agent, output_path)
        
        print("\nOperación completada correctamente.")
        print(f"Puedes usar este modelo corregido ejecutando:")
        print(f"python SnakeNeuralAdvanced.py --episodes 100 --save-path {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error durante la corrección del modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corregir dimensiones en un modelo de SnakeNN")
    parser.add_argument("--model", default="snake_agent.pth", help="Ruta al modelo a corregir")
    parser.add_argument("--output", default=None, help="Ruta para guardar el modelo corregido")
    
    args = parser.parse_args()
    
    fix_model_dimensions(args.model, args.output) 