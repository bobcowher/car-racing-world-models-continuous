import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # prevent Qt thread-affinity warnings
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gymnasium as gym
from agent import Agent


SCALE     = 5    # display scale factor (96 → 480)
FPS       = 1    # playback speed (1 second per frame)
GAP       = 4    # pixel gap between real and imagined panels
MAX_STEPS = 200  # steps per rollout


def embed_to_frame(embed, world_model):
    """Decode a latent embed to a uint8 HWC numpy frame."""
    with torch.no_grad():
        img = world_model.decode(embed)  # (1, C, H, W) in [0,1]
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def obs_to_frame(obs_tensor):
    """Convert a (C, H, W) uint8 tensor to HWC numpy."""
    return obs_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def annotate(frame, label, reward=None):
    frame = frame.copy()
    cv2.putText(frame, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (255, 255, 255), 1, cv2.LINE_AA)
    if reward is not None:
        r_str = f"r={reward:+.2f}"
        cv2.putText(frame, r_str, (4, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 255, 100), 1, cv2.LINE_AA)
    return frame


def scale_up(frame, factor):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def make_display(real_frame, imag_frame, step, real_reward, imag_reward):
    real_ann  = annotate(real_frame,  f"Real  step {step}", reward=real_reward)
    imag_ann  = annotate(imag_frame,  f"Imag  step {step}", reward=imag_reward)
    real_big  = scale_up(real_ann, SCALE)
    imag_big  = scale_up(imag_ann, SCALE)
    gap       = np.zeros((real_big.shape[0], GAP, 3), dtype=np.uint8)
    combined  = np.concatenate([real_big, gap, imag_big], axis=1)
    # OpenCV expects BGR
    return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

    agent = Agent(env=env)

    for path in ["checkpoints/world_model.pt", "checkpoints/q_model.pt"]:
        if not os.path.exists(path):
            print(f"ERROR: checkpoint not found: {path}")
            env.close()
            sys.exit(1)

    agent.load()
    agent.world_model.eval()
    agent.q_model.eval()

    n_actions = agent.world_model.n_actions

    def get_initial_embed():
        obs, _ = env.reset()
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
        obs_tensor = agent.process_observation(obs)
        obs_norm   = obs_tensor.float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            embed = agent.world_model.encode(obs_norm).squeeze(1)
        return obs_tensor, embed

    cv2.namedWindow("Imagination Rollout", cv2.WINDOW_AUTOSIZE)
    delay_ms = int(1000 / FPS)

    obs_tensor, current_embed = get_initial_embed()
    # Show the real starting frame on both panels before the loop begins
    start_frame = obs_to_frame(obs_tensor)
    cv2.imshow("Imagination Rollout", make_display(start_frame, start_frame, 0, 0.0, 0.0))
    cv2.waitKey(delay_ms)

    step = 0
    print("Running imagination rollout. Press 'q' to quit.")

    try:
        while True:
            with torch.no_grad():
                # === Imagination: action from imagined embed (closed loop) ===
                action_idx    = agent.q_model(current_embed).argmax(dim=1)    # (1,)
                action_onehot = F.one_hot(action_idx, num_classes=n_actions).float()

                next_embed, reward_pred, done_pred = agent.world_model.imagine_step(
                    current_embed, action_onehot
                )
                imag_reward = reward_pred.item()
                imag_done   = done_pred.item() > 0.5

                # === Real env: step with same action ===
                next_obs_raw, real_reward, real_term, real_trunc, _ = env.step(action_idx.item())
                next_obs_tensor = agent.process_observation(next_obs_raw)

                # Decode imagined frame for display
                imag_frame = embed_to_frame(next_embed, agent.world_model)
                real_frame = obs_to_frame(next_obs_tensor)

            panel = make_display(real_frame, imag_frame, step, real_reward, imag_reward)
            cv2.imshow("Imagination Rollout", panel)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                break

            current_embed = next_embed
            step += 1

            if imag_done or real_term or real_trunc or step >= MAX_STEPS:
                print(f"Episode ended at step {step}. Closing in 10 seconds (press 'q' to quit early).")
                if cv2.waitKey(10000) & 0xFF == ord('q'):
                    break
                break

    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
