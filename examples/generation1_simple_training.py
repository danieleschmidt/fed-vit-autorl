#!/usr/bin/env python3
"""Generation 1: Simple Federated ViT-AutoRL Training Example

Demonstrates basic functionality with minimal viable features:
- ViT perception model initialization
- Simple federated learning setup
- Basic training loop with mock data
- Essential error handling
"""

import torch
import torch.nn.functional as F
import numpy as np
from fed_vit_autorl import (
    FederatedVehicleRL,
    ViTPerception,
    FederatedSAC
)
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_driving_data(batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic driving data for demonstration."""
    # Mock camera images (384x384 RGB)
    images = torch.randn(batch_size, 3, 384, 384)
    
    # Mock vehicle states (speed, steering, etc.)
    vehicle_states = torch.randn(batch_size, 16)
    
    # Mock driving actions (steering, throttle, brake, gear)
    actions = torch.randn(batch_size, 4)
    actions = torch.tanh(actions)  # Normalize to [-1, 1]
    
    return images, vehicle_states, actions


def compute_simple_rewards(actions: torch.Tensor, target_speed: float = 0.8) -> torch.Tensor:
    """Compute simple driving rewards."""
    # Reward for maintaining target speed (throttle around target_speed)
    throttle = actions[:, 1]  # Assume throttle is second action
    speed_reward = -torch.abs(throttle - target_speed)
    
    # Penalty for extreme steering
    steering = actions[:, 0]  # Assume steering is first action
    steering_penalty = -torch.abs(steering) * 0.5
    
    # Combine rewards
    total_reward = speed_reward + steering_penalty
    return total_reward


class SimpleVehicleEnvironment:
    """Simplified vehicle environment for Generation 1."""
    
    def __init__(self, num_vehicles: int = 10):
        self.num_vehicles = num_vehicles
        self.current_states = {}
        self.reset()
    
    def reset(self):
        """Reset environment state."""
        for vehicle_id in range(self.num_vehicles):
            self.current_states[vehicle_id] = torch.randn(768)  # ViT embedding size
    
    def step(self, vehicle_id: int, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """Take environment step."""
        # Simple state transition (add some noise)
        noise = torch.randn_like(self.current_states[vehicle_id]) * 0.1
        self.current_states[vehicle_id] += noise
        
        # Compute reward
        reward = compute_simple_rewards(action.unsqueeze(0)).item()
        
        # Simple done condition (random for demo)
        done = np.random.random() < 0.05  # 5% chance of episode end
        
        return self.current_states[vehicle_id], reward, done


def main():
    """Generation 1: Simple federated training demonstration."""
    logger.info("🚀 Starting Generation 1: Simple Fed-ViT-AutoRL Training")
    
    try:
        # Initialize perception model
        perception_model = ViTPerception(
            img_size=384,
            patch_size=16,
            num_classes=1000,
            embed_dim=768,
            depth=6,  # Smaller for Generation 1
            num_heads=12
        )
        logger.info("✅ ViT Perception model initialized")
        
        # Create federated RL system
        fed_rl = FederatedVehicleRL(
            model=perception_model,
            num_vehicles=10,  # Small number for Generation 1
            aggregation="fedavg",
            privacy_mechanism="differential_privacy",
            epsilon=1.0
        )
        logger.info("✅ Federated RL system created")
        
        # Initialize SAC agents for each vehicle
        sac_agents = {}
        for vehicle_id in range(fed_rl.num_vehicles):
            sac_agents[vehicle_id] = FederatedSAC(
                state_dim=768,  # ViT embedding dimension
                action_dim=4,   # steering, throttle, brake, gear
                hidden_dim=128, # Smaller for Generation 1
                device="cpu"    # CPU-only for Generation 1
            )
        
        # Simple environment
        env = SimpleVehicleEnvironment(num_vehicles=fed_rl.num_vehicles)
        
        # Training parameters
        num_rounds = 5  # Small number for Generation 1
        episodes_per_round = 3
        max_steps_per_episode = 20
        
        logger.info("🔄 Starting federated training loop")
        
        for round_idx in range(num_rounds):
            logger.info(f"--- Round {round_idx + 1}/{num_rounds} ---")
            
            round_rewards = []
            
            # Each vehicle trains locally
            for vehicle_id in range(fed_rl.num_vehicles):
                vehicle_rewards = []
                
                for episode in range(episodes_per_round):
                    env.reset()
                    episode_reward = 0
                    
                    # Collect episode data
                    states, actions, rewards, next_states, dones = [], [], [], [], []
                    
                    for step in range(max_steps_per_episode):
                        # Get current state
                        current_state = env.current_states[vehicle_id]
                        
                        # Select action using SAC
                        action = sac_agents[vehicle_id].act(current_state)
                        
                        # Take environment step
                        next_state, reward, done = env.step(vehicle_id, action)
                        
                        # Store transition
                        states.append(current_state)
                        actions.append(action.squeeze(0))
                        rewards.append(reward)
                        next_states.append(next_state)
                        dones.append(float(done))
                        
                        episode_reward += reward
                        
                        if done:
                            break
                    
                    # Update SAC if we have enough data
                    if len(states) >= 10:  # Minimum batch size
                        # Convert to tensors
                        states_tensor = torch.stack(states)
                        actions_tensor = torch.stack(actions)
                        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                        next_states_tensor = torch.stack(next_states)
                        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
                        
                        # Update SAC
                        update_info = sac_agents[vehicle_id].update(
                            states_tensor,
                            actions_tensor,
                            rewards_tensor,
                            next_states_tensor,
                            dones_tensor
                        )
                        
                        if episode == 0:  # Log first episode info
                            logger.info(f"Vehicle {vehicle_id} - Actor Loss: {update_info['actor_loss']:.4f}")
                    
                    vehicle_rewards.append(episode_reward)
                
                avg_reward = np.mean(vehicle_rewards)
                round_rewards.append(avg_reward)
                logger.info(f"Vehicle {vehicle_id}: Avg Reward = {avg_reward:.2f}")
            
            # Simple federated aggregation (average model states)
            if round_idx < num_rounds - 1:  # Don't aggregate on last round
                logger.info("🔄 Performing federated aggregation")
                
                # Collect all model states
                all_states = []
                for vehicle_id in range(fed_rl.num_vehicles):
                    state = sac_agents[vehicle_id].get_model_state()
                    all_states.append(state)
                
                # Simple averaging (Generation 1 approach)
                if all_states:
                    averaged_state = {}
                    for key in all_states[0].keys():
                        averaged_state[key] = {}
                        for param_name in all_states[0][key].keys():
                            param_sum = sum(state[key][param_name] for state in all_states)
                            averaged_state[key][param_name] = param_sum / len(all_states)
                    
                    # Distribute averaged model to all vehicles
                    for vehicle_id in range(fed_rl.num_vehicles):
                        sac_agents[vehicle_id].set_model_state(averaged_state)
                    
                    logger.info("✅ Federated aggregation completed")
            
            # Round summary
            avg_round_reward = np.mean(round_rewards)
            logger.info(f"Round {round_idx + 1} Average Reward: {avg_round_reward:.2f}")
        
        logger.info("🎉 Generation 1 training completed successfully!")
        
        # Final evaluation
        logger.info("📊 Final Evaluation")
        total_final_reward = 0
        
        for vehicle_id in range(min(3, fed_rl.num_vehicles)):  # Test first 3 vehicles
            env.reset()
            eval_reward = 0
            
            for step in range(10):  # Short evaluation
                state = env.current_states[vehicle_id]
                action = sac_agents[vehicle_id].act(state, deterministic=True)
                next_state, reward, done = env.step(vehicle_id, action)
                eval_reward += reward
                
                if done:
                    break
            
            total_final_reward += eval_reward
            logger.info(f"Vehicle {vehicle_id} Final Eval Reward: {eval_reward:.2f}")
        
        avg_final_reward = total_final_reward / min(3, fed_rl.num_vehicles)
        logger.info(f"🏆 Average Final Evaluation Reward: {avg_final_reward:.2f}")
        
        # Save model checkpoint (Generation 1)
        checkpoint = {
            'fed_rl_config': {
                'num_vehicles': fed_rl.num_vehicles,
                'aggregation': fed_rl.aggregation,
                'epsilon': fed_rl.epsilon
            },
            'sac_configs': {
                'state_dim': 768,
                'action_dim': 4,
                'hidden_dim': 128
            },
            'training_results': {
                'final_avg_reward': avg_final_reward,
                'num_rounds': num_rounds
            }
        }
        
        # Save first vehicle's model as example
        checkpoint['example_model_state'] = sac_agents[0].get_model_state()
        
        torch.save(checkpoint, 'generation1_checkpoint.pth')
        logger.info("💾 Checkpoint saved as generation1_checkpoint.pth")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation 1 training failed: {str(e)}")
        logger.error("This indicates missing dependencies or configuration issues")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Generation 1 demonstration completed successfully!")
        exit(0)
    else:
        print("❌ Generation 1 demonstration failed!")
        exit(1)