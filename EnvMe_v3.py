import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [%(levelname)s]:%(message)s', datefmt='%d/%m/%Y_%H:%M:%S', filename='EnvMe.log', encoding='utf-8', level=logging.INFO)

import time

class EnvNetwork(gym.Env):
    def __init__(self, attack_type: int, debug: bool = False):
        self.services = {
            'ftp': {
                'probability': 0.0987,
                'cost': 2
            },
            'ssh': {
                'probability': 0.0729,
                'cost': 3
            },
            'telnet': {
                'probability': 0.0395,
                'cost': 1
            },
            'http': {
                'probability': 0.7889,
                'cost': 2
            },
            'smtp': {
                'probability': 0.0455,
                'cost': 4
            },
            'dns': {
                'probability': 0.0533,
                'cost': 2
            },
        }
        self.num_services = len(self.services.keys())
        self.nodes = {
            'Nr': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': True,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': True},
                'links': {'Nr': True, 'FWe': True,'FW1': False, 'FW2': False, 'Nc': False, 'N1': False, 'N2': False, 'U0': False, 'U1': False, 'U2': False, 'U3': False, 'U4': False, 'U5': False,}                
            },
            'FWe': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': False,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': True},
                'links': {'Nr': True, 'FWe': True, 'FW1': False, 'FW2': False, 'Nc': True, 'N1': False, 'N2': False, 'U0': False, 'U1': False, 'U2': False, 'U3': False, 'U4': False, 'U5': False,}            
            },
             'FW1': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': False,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': False},
                'links': {'Nr': False, 'FWe': False, 'FW1': True, 'FW2': False, 'Nc': False, 'N1': True, 'N2': False, 'U0': True, 'U1': True, 'U2': True, 'U3': False, 'U4': False, 'U5': False,}
            },
            'FW2': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': False,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': False},
                'links': {'Nr': False, 'FWe': False, 'FW1': False, 'FW2': True, 'Nc': False, 'N1': False, 'N2': True, 'U0': False, 'U1': False, 'U2': False, 'U3': True, 'U4': True, 'U5': True,}
            },
            'Nc': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': True,'ssh': True, 'telnet': False, 'http': True, 'smtp': True, 'dns': True},
                'links': {'Nr': False, 'FWe': True, 'FW1': False, 'FW2': False, 'Nc': True, 'N1': True, 'N2': True, 'U0': False, 'U1': False, 'U2': False, 'U3': False, 'U4': False, 'U5': False,}
            }, 
            'N1': {
                'important': True, 
                'compromised': False, 
                'services': {'ftp': True,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': True},
                'links': {'Nr': False, 'FWe': False, 'FW1': False, 'FW2': False, 'Nc': True, 'N1': True, 'N2': False, 'U0': True, 'U1': True, 'U2': True, 'U3': False, 'U4': False, 'U5': False,}
            },      
            'N2': {
                'important': True, 
                'compromised': False, 
                'services': {'ftp': True,'ssh': True, 'telnet': True, 'http': True, 'smtp': True, 'dns': True},
                'links': {'Nr': False, 'FWe': False, 'FW1': False, 'FW2': False, 'Nc': True, 'N1': False, 'N2': True, 'U0': False, 'U1': False, 'U2': False, 'U3': True, 'U4': True, 'U5': True,}
            },
            'U0': {
                'important': True, 
                'compromised': False, 
                'services': {'ftp': False,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': True},
                'links': {'Nr': False, 'FWe': False, 'FW1': True, 'FW2': False, 'Nc': False, 'N1': True, 'N2': False, 'U0': True, 'U1': True, 'U2': True, 'U3': False, 'U4': False, 'U5': False,}
            },
            'U1': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': False,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': True},
                'links': {'Nr': False, 'FWe': False, 'FW1': True, 'FW2': False, 'Nc': False, 'N1': True, 'N2': False, 'U0': True, 'U1': True, 'U2': True, 'U3': False, 'U4': False, 'U5': False,}
            },
            'U2': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': True,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': False},
                'links': {'Nr': False, 'FWe': False, 'FW1': True, 'FW2': False, 'Nc': False, 'N1': True, 'N2': False, 'U0': True, 'U1': True, 'U2': True, 'U3': False, 'U4': False, 'U5': False,}
            },
            'U3': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': True,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': True},
                'links': {'Nr': False, 'FWe': False, 'FW1': False, 'FW2': True, 'Nc': False, 'N1': False, 'N2': True, 'U0': False, 'U1': False, 'U2': False, 'U3': True, 'U4': True, 'U5': True,}
            },
            'U4': {
                'important': True, 
                'compromised': False, 
                'services': {'ftp': False,'ssh': True, 'telnet': False, 'http': True, 'smtp': True, 'dns': True},
                'links': {'Nr': False, 'FWe': False, 'FW1': False, 'FW2': True, 'Nc': False, 'N1': False, 'N2': True, 'U0': False, 'U1': False, 'U2': False, 'U3': True, 'U4': True, 'U5': True,}
            },
            'U5': {
                'important': False, 
                'compromised': False, 
                'services': {'ftp': True,'ssh': True, 'telnet': False, 'http': True, 'smtp': False, 'dns': True},
                'links': {'Nr': False, 'FWe': False, 'FW1': False, 'FW2': True, 'Nc': False, 'N1': False, 'N2': True, 'U0': False, 'U1': False, 'U2': False, 'U3': True, 'U4': True, 'U5': True,}
            },
        }
        self.num_nodes = len(self.nodes.keys())
        self.critical_nodes = ['U0', 'U4']
        self.attack_type = attack_type
        self.attack_action = attack_action
        self.debug = debug
        self.hacked = False
        self.detected = False
        self.status = ''
        self.done = False
        self.detection_trayectory = ['']
        self.detection_node = ''
        self.detection_protocol = ''
        self.current_node = 'Nr'
        self.attack_path = [self.current_node]
        self.total_reward = 0
        self.iteration = 0
        self.observation_space = gym.spaces.Box(low=np.zeros((len(self.nodes), len(self.services))), high=np.ones((len(self.nodes), len(self.services))), dtype=np.int32)
        self.state = self.observation_space.low
        self.hacked_port = ''
        self.action_space = gym.spaces.Discrete(len(self.nodes)*len(self.services))  # Blocking position
        self.data = pd.DataFrame()
        self.iteration_time = time.time()
        self.PENALTY_INSUFFICIENT_SUPERVISION = -75.0
    def _check_inadequate_supervision(self, action: int) -> float:
        compromised_critical_nodes = [name for name in self.critical_nodes if self.nodes[name]['compromised']]
        if compromised_critical_nodes:
            # Índice del nodo sobre el que el defensor intentó actuar
            node_index_acted = action // self.num_services
            node_name_acted = self.node_names[node_index_acted]
            # Si el crítico está comprometido Y la acción NO fue sobre U0 o U4
            if node_name_acted not in self.critical_nodes:
                if self.debug:
                    logger.warning(f"-> Penalización: Nodo(s) crítico(s) comprometido(s). Acción defensiva ignorada en {node_name_acted}")
                return self.PENALTY_INSUFFICIENT_SUPERVISION
        return 0.0
    def _check_lack_of_segmentation(self) -> float:
        """
        Penaliza si el atacante logra saltar el FW e ingresar al Core (Nc), 
        que es la puerta de entrada a todos los servicios internos (Falla crítica de segmentación).
        """
        PENALTY_LACK_OF_SEGMENTATION = -100.0 
        if len(self.attack_path) >= 2:
            last_node = self.attack_path[-2]
            current_node = self.attack_path[-1]
            if last_node == 'FWe' and current_node == 'Nc' and self.nodes[current_node]['compromised']:
                if self.debug:
                    logger.warning(f"-> Penalización: Falla crítica de segmentación detectada (FW -> Nc)")
                return PENALTY_LACK_OF_SEGMENTATION
        return 0.0
    #def _check_overreaction(self, action: int) -> float:
    #    """
    #    Penaliza la sobrerreacción/ineficiencia. Penalizamos si la acción es 
    #    bloquear el protocolo de menor riesgo ('telnet') en un nodo NO comprometido.
    #    """
    #    PENALTY_OVERREACTION = -20.0
    #    telnet_index = self.service_names.index('telnet')
    #    target_service_idx = action % self.num_services
    #
    #    if target_service_idx == telnet_index:
    #        # Solo penalizar si la acción no está en un nodo ya comprometido (donde la acción es necesaria)
    #        node_index = action // self.num_services
    #        node_name = self.node_names[node_index]
    #
    #        if not self.nodes[node_name]['compromised']:
    #            if self.debug:
    #                logger.info(f"-> Penalización: Sobrerreacción, acción sobre servicio de baja probabilidad ('telnet') en nodo seguro")
    #            return PENALTY_OVERREACTION
    #
    #    return 0.0
    def get_service_with_highest_probability(self, configured_services):
        service_with_highest_probability = None
        highest_probability = 0.0
        for service in configured_services:
            probability = self.services[service]['probability']
            if probability > highest_probability:
                service_with_highest_probability = service
                highest_probability = probability
        return service_with_highest_probability
    def reset(self, seed: int = 0):
        if self.iteration != 0:
            new_iteration = {
                "steps": self.iteration, 
                "rewards": self.total_reward,
                "last_node": ["U0/U4 Compromised" if self.hacked else self.detection_node], # Nodos críticos
                "protocol": [self.hacked_port if self.hacked else self.detection_protocol], 
                "time": '{:.4f}'.format(self.iteration_time),
                "status": self.status
            }
            # Nota: 'self.data' y 'self.data._append' requieren que la librería pandas esté disponible.
            self.data = self.data._append(new_iteration, ignore_index=True)
        self.iteration_time = time.time()
        if self.debug:
            logger.info("Starting the environment!")
        self.hacked = False
        self.detected = False
        self.status = ''
        self.done = False
        self.detection_trayectory = ['']
        self.detection_node = ''
        self.detection_protocol = ''
        self.current_node = 'Nr' 
        self.attack_path = [self.current_node]
        self.total_reward = 0
        self.iteration = 0
        self.observation_space = gym.spaces.Box(
            low=np.zeros((len(self.nodes), len(self.services))), 
            high=np.ones((len(self.nodes), len(self.services))), 
            dtype=np.int32
        )
        self.state = self.observation_space.low
        self.hacked_port = ''
        return self.observation_space.low, {}
    def step(self, action):
        reward = 0
        self.iteration += 1
        if self.debug: logger.info(f"Action: {action}")
        # --- CÁLCULO DE PENALIZACIONES (REWARD SHAPING) ---
        penalty_supervision = self._check_inadequate_supervision(action)
        penalty_segmentation = self._check_lack_of_segmentation()
        #penalty_overreaction = self._check_overreaction(action)    
        shaping_reward = penalty_supervision + penalty_segmentation #+ penalty_overreaction  
        # --- FIN CÁLCULO DE PENALIZACIONES ---
        position = [ action // self.num_nodes , action % self.num_services ] # Convierte la acción discreta a una posición [nodo_idx, servicio_idx] en la matriz de estado.
        if position == self.detection_trayectory[-1] and self.attack_path[-1] == list(self.nodes)[position[0]]:
            # Lógica de Detección/Bloqueo del Agente Defensor
            self.detection_node = list(self.nodes)[position[0]]
            self.detection_protocol = list(self.services)[position[1]]
            self.detected = True
            # reward += 20 # Línea comentada, el premio base por detección debe ser grande.
            if self.debug: logger.info(f"Attack detected in the node {self.detection_node} to {self.detection_protocol} protocol")
        else:
            # Si el agente defensor no detecta el ataque
            self.detection_trayectory.append(position)
            #if self.observation_space.low[position][position[1]]!= 0:
            #    self.observation_space.low[position][position[1]] -= 1
            #    reward += 5 # Recompensa base
            if self.observation_space.low[position[0],position[1]] != 0:
                self.observation_space.low[position[0],position[1]] -= 1
                reward += 5 # Recompensa base
            else:
                reward += 1 # Recompensa base
            if self.debug: logger.info(f"Detection trayectory: {self.detection_trayectory}")
        # Simulación de la Acción del Atacante (si attack action es 1 y no ha sido detectado)
        if self.attack_action == 1 and not self.detected:  # trying to exploit
            # Selecciona un nodo vecino aleatorio
            neighbors = [node for node, linked in self.nodes[self.current_node]['links'].items() if linked]
            next_node = random.choice(neighbors)
            # Selecciona el servicio configurado con la mayor probabilidad de éxito en la explotación
            configured_services = [service for service, enabled in self.nodes[next_node]['services'].items() if enabled]
            if configured_services:
                service = self.get_service_with_highest_probability(configured_services)
                # Intenta explotar el servicio
                if random.random() < self.services[service]['probability']:
                    self.observation_space.low[list(self.nodes).index(next_node)][list(self.services).index(service)] += 1
                    if self.debug: logger.info(f"Exploiting {service} on {next_node}...")
                    # Si el nivel de explotación alcanza el 'cost' del servicio, el nodo es comprometido
                    if self.observation_space.low[list(self.nodes).index(next_node)][list(self.services).index(service)] >= self.services[service]['cost']:
                        self.nodes[next_node]['compromised'] = True
                        self.current_node = next_node # El atacante se mueve al nodo comprometido
                        self.attack_path.append(next_node)
                        if self.debug: logger.info(f"Moving to {next_node}...")
                        reward += -1 # Recompensa base negativa por el avance del atacante
                    else:
                        reward += 0
                else:
                    if self.debug: logger.info(f"Failed to exploit {service} on {next_node}...")
                    reward += 0
            else:
                reward += 0
        else:
            reward += 0
        # Lógica de Terminación del Episodio (Hacked o Detected)
        if self.current_node == ('U0','U4'):
            reward += -100 # Penalización grande por compromiso del nodo crítico
            self.hacked = True
            self.hacked_port = service
            self.status = "hacked"
            self.done = True
        if self.detected:
            reward += 100 # Recompensa grande por detección exitosa
            self.status = "detected"
            self.done = True
        # APLICACIÓN FINAL DEL REWARD SHAPING
        reward += shaping_reward 
        # Actualización de estado y retorno
        if self.detected or self.hacked:
            self.iteration_time -= time.time()
            self.iteration_time *= -1
        self.total_reward += reward
        self.state = self.observation_space.low
        return self.observation_space.low, reward, self.done, self.hacked, {}

# Función para entrenar y evaluar los algoritmos
def train_and_evaluate(model, eval_env, log_path, num_episodes, eval_frequency):
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path, eval_freq=eval_frequency,
                                 log_path=log_path, deterministic=True, render=False)
    # 2. Entrenar el modelo
    logger.info(f"Iniciando entrenamiento de {model.__class__.__name__}...")
    # model.learn(total_timesteps=num_episodes, callback=eval_callback)
    model.learn(total_timesteps=num_episodes, callback=eval_callback)
    logger.info(f"Entrenamiento de {model.__class__.__name__} finalizado.")
    # 3. Generar y guardar resultados
    try:
        # Guardar datos brutos de la simulación
        eval_env.env.data.to_csv(f"{log_path}/samples-{model.__class__.__name__}.csv", index=False)
        df = eval_env.env.data.dropna()
        if df.empty:
            logger.warning(f"Advertencia: El DataFrame de resultados para {model.__class__.__name__} está vacío después de limpiar.")
            return
        # Gráfico 1: Ataque vs. Defensa (Estados Finales)
        plt.figure(figsize=(10, 6))
        ax = df['status'].value_counts().plot(kind='bar', title=f'Resultados del Episodio - {model.__class__.__name__}')
        plt.xticks(rotation=0)
        for container in ax.containers:
            ax.bar_label(container)
        plt.savefig(f'{log_path}/AttackervsDefender-{model.__class__.__name__}.png')
        plt.close() # Cierra la figura para liberar memoria
        # Gráfico 2: Acciones por Iteración (Eficiencia)
        plt.figure(figsize=(10, 6))
        h = df['steps'].loc[df['status'] == 'hacked']
        d = df['steps'].loc[df['status'] == 'detected']
        h.plot(label="Hacked (Atacante Ganó)", color='r', marker = 'o')
        d.plot(label="Detected (Defensor Ganó)", color='g', marker = 'x')
        plt.title(f"Pasos/Acciones por Episodio - {model.__class__.__name__}")
        plt.xlabel("Muestra de Episodio")
        plt.ylabel("Número de Pasos (Acciones)")
        plt.legend()
        plt.savefig(f'{log_path}/ActionsPerIteration-{model.__class__.__name__}.png')
        plt.close() # Cierra la figura
    except Exception as e:
        logger.error(f"Error al procesar resultados para {model.__class__.__name__}: {e}")
        pass

if __name__ == '__main__':
    logger.info("Iniciando el entrenamiento de Agentes Defensivos!")
    # 1. Configuración
    log_path_ppo = './RL_Training_Results_PPO'
    log_path_dqn = './RL_Training_Results_DQN'
    num_episodes = 100000  # Reducido para una prueba más rápida (ajusta a 500000)
    eval_frequency = 5000 # Frecuencia de evaluación (cada 5000 pasos)
    # 4. Entrenar y evaluar PPO
    logger.info("--- Comenzando PPO ---")
    env_ppo = EnvNetwork(attack_type=1, debug=False)
    # Envuelve el entorno para registrar métricas
    env_ppo = Monitor(env_ppo, log_path_ppo)
    # "MlpPolicy": Red Neuronal Multicapa (MLP) - La política por defecto
    #ppo_model = PPO("MlpPolicy", env_ppo, verbose=1, device="auto")
    ppo_model = PPO("MlpPolicy", env_ppo, verbose=1, device="cpu")
    start_train_time = time.time()
    train_and_evaluate(ppo_model, env_ppo, log_path_ppo, num_episodes, eval_frequency)
    end_train_time = time.time() - start_train_time
    logger.info(f"Modelo PPO entrenado en: {end_train_time:.2f} segundos\n")
    # 5. Entrenar y evaluar DQN
    logger.info("--- Comenzando DQN ---")
    env_dqn = EnvNetwork(attack_type=1, debug=False)
    env_dqn = Monitor(env_dqn, log_path_dqn)
    #dqn_model = DQN("MlpPolicy", env_dqn, verbose=1, device="auto")
    dqn_model = DQN("MlpPolicy", env_dqn, verbose=1, device="cpu")
    start_train_time = time.time()
    train_and_evaluate(dqn_model, env_dqn, log_path_dqn, num_episodes, eval_frequency)
    end_train_time = time.time() - start_train_time
    logger.info(f"Modelo DQN entrenado en: {end_train_time:.2f} segundos\n")
