"""CARLA simulation environment for federated autonomous driving."""

import logging
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class CARLAFederatedEnv:
    """CARLA-based federated learning environment for autonomous vehicles."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town: str = "Town05",
        weather: str = "ClearNoon",
        num_vehicles: int = 20,
        traffic_density: float = 0.3,
        synchronous_mode: bool = True,
        fixed_delta_seconds: float = 0.05,
    ):
        """Initialize CARLA federated environment.
        
        Args:
            host: CARLA server host
            port: CARLA server port
            town: CARLA town/map to use
            weather: Weather conditions
            num_vehicles: Number of federated vehicles
            traffic_density: Background traffic density
            synchronous_mode: Whether to use synchronous simulation
            fixed_delta_seconds: Fixed time step for synchronous mode
        """
        self.host = host
        self.port = port
        self.town = town
        self.weather = weather
        self.num_vehicles = num_vehicles
        self.traffic_density = traffic_density
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        
        # CARLA client and world
        self.client = None
        self.world = None
        self.map = None
        
        # Vehicle management
        self.federated_vehicles = {}
        self.background_vehicles = []
        self.spawn_points = []
        
        # Simulation state
        self.current_step = 0
        self.episode_length = 1000
        
        logger.info(f"Initialized CARLA federated environment: {num_vehicles} vehicles in {town}")
    
    def connect(self) -> bool:
        """Connect to CARLA server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import CARLA (optional dependency)
            import carla
            
            # Connect to CARLA server
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            
            # Load world and map
            self.world = self.client.load_world(self.town)
            self.map = self.world.get_map()
            
            # Set weather
            weather_presets = {
                "ClearNoon": carla.WeatherParameters.ClearNoon,
                "CloudyNoon": carla.WeatherParameters.CloudyNoon,
                "WetNoon": carla.WeatherParameters.WetNoon,
                "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
                "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
                "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
                "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            }
            
            if self.weather in weather_presets:
                self.world.set_weather(weather_presets[self.weather])
            
            # Configure synchronous mode
            if self.synchronous_mode:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_delta_seconds
                self.world.apply_settings(settings)
            
            # Get spawn points
            self.spawn_points = self.map.get_spawn_points()
            random.shuffle(self.spawn_points)
            
            logger.info(f"Connected to CARLA server: {self.town} with {len(self.spawn_points)} spawn points")
            return True
            
        except ImportError:
            logger.error("CARLA package not available. Install with: pip install carla")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
    
    def spawn_federated_vehicles(self) -> Dict[str, Any]:
        """Spawn federated learning vehicles.
        
        Returns:
            Dictionary of spawned vehicle information
        """
        if not self.client or not self.world:
            logger.error("Not connected to CARLA server")
            return {}
        
        import carla
        
        # Get vehicle blueprints
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        
        # Remove bikes and motorcycles for simplicity
        vehicle_blueprints = [bp for bp in vehicle_blueprints 
                             if not (bp.id.endswith('bike') or bp.id.endswith('motorcycle'))]
        
        spawned_vehicles = {}
        
        for i in range(min(self.num_vehicles, len(self.spawn_points))):
            vehicle_id = f"federated_vehicle_{i}"
            
            try:
                # Select random blueprint and spawn point
                blueprint = random.choice(vehicle_blueprints)
                spawn_point = self.spawn_points[i]
                
                # Spawn vehicle
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
                
                # Create vehicle client
                vehicle_client = CARLAVehicleClient(
                    vehicle_id=vehicle_id,
                    carla_vehicle=vehicle,
                    world=self.world,
                )
                
                self.federated_vehicles[vehicle_id] = vehicle_client
                spawned_vehicles[vehicle_id] = {
                    'blueprint': blueprint.id,
                    'spawn_point': spawn_point,
                    'vehicle': vehicle,
                }
                
                logger.debug(f"Spawned federated vehicle {vehicle_id}")
                
            except Exception as e:
                logger.warning(f"Failed to spawn vehicle {vehicle_id}: {e}")
        
        logger.info(f"Spawned {len(spawned_vehicles)} federated vehicles")
        return spawned_vehicles
    
    def spawn_background_traffic(self) -> List[Any]:
        """Spawn background traffic vehicles.
        
        Returns:
            List of spawned background vehicles
        """
        if not self.client or not self.world:
            return []
        
        import carla
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        
        # Calculate number of background vehicles
        num_background = int(len(self.spawn_points) * self.traffic_density)
        num_background = min(num_background, len(self.spawn_points) - self.num_vehicles)
        
        background_vehicles = []
        used_spawn_points = self.spawn_points[:self.num_vehicles]  # Reserved for federated vehicles
        available_spawn_points = self.spawn_points[self.num_vehicles:]
        
        for i in range(min(num_background, len(available_spawn_points))):
            try:
                blueprint = random.choice(vehicle_blueprints)
                spawn_point = available_spawn_points[i]
                
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
                vehicle.set_autopilot(True)  # Enable autopilot for background traffic
                
                background_vehicles.append(vehicle)
                
            except Exception as e:
                logger.debug(f"Failed to spawn background vehicle {i}: {e}")
        
        self.background_vehicles = background_vehicles
        logger.info(f"Spawned {len(background_vehicles)} background vehicles")
        return background_vehicles
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the simulation environment.
        
        Returns:
            Dictionary of initial observations for each vehicle
        """
        # Clean up existing vehicles
        self.cleanup()
        
        # Spawn new vehicles
        self.spawn_federated_vehicles()
        self.spawn_background_traffic()
        
        # Reset simulation state
        self.current_step = 0
        
        # Get initial observations
        observations = {}
        for vehicle_id, vehicle_client in self.federated_vehicles.items():
            obs = vehicle_client.get_observation()
            observations[vehicle_id] = obs
        
        # Tick simulation
        if self.synchronous_mode:
            self.world.tick()
        
        return observations
    
    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Dict]]:
        """Execute one simulation step.
        
        Args:
            actions: Dictionary mapping vehicle_id to action array
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # Apply actions to vehicles
        for vehicle_id, action in actions.items():
            if vehicle_id in self.federated_vehicles:
                vehicle_client = self.federated_vehicles[vehicle_id]
                vehicle_client.apply_action(action)
        
        # Tick simulation
        if self.synchronous_mode:
            self.world.tick()
        else:
            time.sleep(self.fixed_delta_seconds)
        
        # Get observations, rewards, and done flags
        for vehicle_id, vehicle_client in self.federated_vehicles.items():
            obs = vehicle_client.get_observation()
            reward = vehicle_client.compute_reward()
            done = vehicle_client.is_done() or self.current_step >= self.episode_length
            info = vehicle_client.get_info()
            
            observations[vehicle_id] = obs
            rewards[vehicle_id] = reward
            dones[vehicle_id] = done
            infos[vehicle_id] = info
        
        self.current_step += 1
        
        return observations, rewards, dones, infos
    
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        # Destroy federated vehicles
        for vehicle_client in self.federated_vehicles.values():
            vehicle_client.cleanup()
        self.federated_vehicles.clear()
        
        # Destroy background vehicles
        for vehicle in self.background_vehicles:
            try:
                vehicle.destroy()
            except Exception as e:
                logger.debug(f"Error destroying background vehicle: {e}")
        self.background_vehicles.clear()
    
    def close(self) -> None:
        """Close CARLA environment."""
        self.cleanup()
        
        # Reset synchronous mode
        if self.world and self.synchronous_mode:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except Exception as e:
                logger.debug(f"Error resetting synchronous mode: {e}")


class CARLAVehicleClient:
    """Individual vehicle client for CARLA simulation."""
    
    def __init__(
        self,
        vehicle_id: str,
        carla_vehicle: Any,
        world: Any,
        camera_resolution: Tuple[int, int] = (800, 600),
    ):
        """Initialize CARLA vehicle client.
        
        Args:
            vehicle_id: Unique vehicle identifier
            carla_vehicle: CARLA vehicle actor
            world: CARLA world object
            camera_resolution: Camera resolution (width, height)
        """
        self.vehicle_id = vehicle_id
        self.vehicle = carla_vehicle
        self.world = world
        self.camera_resolution = camera_resolution
        
        # Sensors
        self.camera = None
        self.camera_data = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        # State tracking
        self.collision_occurred = False
        self.lane_invasion_occurred = False
        self.previous_location = None
        self.total_distance = 0.0
        
        # Setup sensors
        self._setup_sensors()
        
        logger.debug(f"Initialized CARLA vehicle client: {vehicle_id}")
    
    def _setup_sensors(self) -> None:
        """Setup vehicle sensors."""
        try:
            import carla
            
            blueprint_library = self.world.get_blueprint_library()
            
            # Camera sensor
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.camera_resolution[0]))
            camera_bp.set_attribute('image_size_y', str(self.camera_resolution[1]))
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),  # Front of vehicle, elevated
                carla.Rotation(pitch=0.0)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            self.camera.listen(self._on_camera_data)
            
            # Collision sensor
            collision_bp = blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.collision_sensor.listen(self._on_collision)
            
            # Lane invasion sensor
            lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
            self.lane_invasion_sensor = self.world.spawn_actor(
                lane_invasion_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.lane_invasion_sensor.listen(self._on_lane_invasion)
            
        except Exception as e:
            logger.error(f"Failed to setup sensors for {self.vehicle_id}: {e}")
    
    def _on_camera_data(self, image) -> None:
        """Callback for camera data."""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # RGBA
        array = array[:, :, :3]  # Remove alpha channel
        self.camera_data = array
    
    def _on_collision(self, event) -> None:
        """Callback for collision events."""
        self.collision_occurred = True
        logger.warning(f"Collision detected for vehicle {self.vehicle_id}")
    
    def _on_lane_invasion(self, event) -> None:
        """Callback for lane invasion events."""
        self.lane_invasion_occurred = True
        logger.debug(f"Lane invasion detected for vehicle {self.vehicle_id}")
    
    def get_observation(self) -> np.ndarray:
        """Get current observation from vehicle sensors.
        
        Returns:
            Observation array (camera image)
        """
        if self.camera_data is not None:
            # Normalize pixel values to [0, 1]
            return self.camera_data.astype(np.float32) / 255.0
        else:
            # Return black image if no camera data available
            return np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.float32)
    
    def apply_action(self, action: np.ndarray) -> None:
        """Apply action to vehicle.
        
        Args:
            action: Action array [throttle, steer, brake] in range [-1, 1]
        """
        try:
            import carla
            
            throttle = max(0.0, float(action[0]))  # [0, 1]
            steer = float(action[1])  # [-1, 1]
            brake = max(0.0, -float(action[0])) if action[0] < 0 else 0.0  # [0, 1]
            
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
            
            self.vehicle.apply_control(control)
            
        except Exception as e:
            logger.error(f"Failed to apply action to vehicle {self.vehicle_id}: {e}")
    
    def compute_reward(self) -> float:
        """Compute reward for current state.
        
        Returns:
            Scalar reward value
        """
        reward = 0.0
        
        # Get vehicle state
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h
        
        location = self.vehicle.get_location()
        
        # Speed reward (encourage appropriate speed)
        target_speed = 50.0  # km/h
        speed_reward = 1.0 - abs(speed - target_speed) / target_speed
        reward += speed_reward * 0.3
        
        # Distance reward
        if self.previous_location is not None:
            distance = location.distance(self.previous_location)
            self.total_distance += distance
            reward += distance * 0.1  # Reward for moving forward
        
        self.previous_location = location
        
        # Safety penalties
        if self.collision_occurred:
            reward -= 10.0
            self.collision_occurred = False  # Reset flag
        
        if self.lane_invasion_occurred:
            reward -= 1.0
            self.lane_invasion_occurred = False  # Reset flag
        
        return reward
    
    def is_done(self) -> bool:
        """Check if episode is done for this vehicle.
        
        Returns:
            True if episode should end, False otherwise
        """
        # End episode on collision
        if self.collision_occurred:
            return True
        
        # End episode if vehicle is stuck (very low speed for extended time)
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        if speed < 0.1:  # Less than 0.1 m/s
            return True
        
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional information about vehicle state.
        
        Returns:
            Dictionary with vehicle state information
        """
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
        
        location = self.vehicle.get_location()
        rotation = self.vehicle.get_transform().rotation
        
        return {
            'speed_kmh': speed,
            'location': (location.x, location.y, location.z),
            'rotation': (rotation.pitch, rotation.yaw, rotation.roll),
            'total_distance': self.total_distance,
            'collision_occurred': self.collision_occurred,
            'lane_invasion_occurred': self.lane_invasion_occurred,
        }
    
    def cleanup(self) -> None:
        """Clean up vehicle and sensors."""
        try:
            if self.camera:
                self.camera.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.lane_invasion_sensor:
                self.lane_invasion_sensor.destroy()
            if self.vehicle:
                self.vehicle.destroy()
        except Exception as e:
            logger.debug(f"Error during cleanup for vehicle {self.vehicle_id}: {e}")