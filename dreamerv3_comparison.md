# Architectural Comparison: car-racing-dreamerv3 vs. DreamerV3

*Generated 2026-05-17*

---

## Overview

This document compares the current `car-racing-dreamerv3` implementation against the
DreamerV3 algorithm (Hafner et al., 2023, "Mastering Diverse Domains with World Models").
The goal is to catalog structural differences and assess the tradeoffs of each approach.

---

## 1. World Model Architecture

| Aspect | car-racing-dreamerv3 | DreamerV3 |
|---|---|---|
| **State representation** | Deterministic continuous vector (1024-dim CNN embedding) | Two-part state: deterministic recurrent `h_t` (GRU) + stochastic categorical `z_t` |
| **Recurrence** | None — each embedding computed independently from a single frame | GRU-based: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`, carries temporal memory across the episode |
| **Stochasticity** | None — encoder is fully deterministic | Categorical latent `z_t` sampled from a learned posterior (encoder) and prior (dynamics). 32 categories × 32 classes = 1024 dims total |
| **Dynamics model** | Feedforward MLP: `embed + action → next_embed` | Two distributions: prior `p(z_t | h_t)` (prediction only) and posterior `q(z_t | h_t, x_t)` (uses actual observation). GRU transitions `h_t` causally. |
| **Training data shape** | Individual transitions `(s, a, r, s', d)` | Sequential chunks (64 timesteps) — required for the recurrent state |
| **Dynamics supervision** | Direct MSE between predicted and actual next embedding | KL divergence between prior and posterior — no direct target needed |

---

## 2. World Model Loss Functions

| Loss | car-racing-dreamerv3 | DreamerV3 |
|---|---|---|
| **Reconstruction** | L1 + 0.2×SSIM + 0.1×gradient loss | Log-likelihood (effectively MSE for a Gaussian decoder) |
| **Dynamics** | MSE(`pred_next_embed`, `actual_next_embed`) | KL(posterior ∥ prior) with **free bits** (min 1 bit/dim) and **KL balancing** (0.8 weight on prior, 0.2 on posterior) |
| **Reward** | MSE on raw rewards | MSE on **symlog-transformed** rewards: `sign(x) · ln(|x| + 1)` |
| **Done / continue** | Binary cross-entropy | Binary cross-entropy (same) |
| **Combined weighting** | `1.0·recon + 1.0·dyn + 2.0·reward + 0.5·done` | `~1.0·recon + 1.0·KL + 1.0·reward + 1.0·continue` |

**Note on the KL component.** In DreamerV3 the KL loss does double duty: it trains the
dynamics model (prior) and regularizes the encoder (posterior) simultaneously. The current
implementation uses a direct supervised signal instead, which is simpler but loses
uncertainty modeling. Free bits prevent early KL collapse by enforcing a minimum information
content per latent dimension.

---

## 3. Actor-Critic

| Aspect | car-racing-dreamerv3 | DreamerV3 |
|---|---|---|
| **Algorithm** | SAC (Soft Actor-Critic) with twin critics | Simple AC — actor maximizes value, no twin critics, no entropy temperature |
| **Training data source** | MixedSampler: 75% real replay, 25% imagined (current run) | 100% imagined rollouts — no real data used for AC training |
| **Return estimation** | 1-step Bellman: `r + γ · V(s')` | **λ-returns** (λ = 0.95): blends multi-step Monte Carlo with bootstrapped value |
| **Value normalization** | None | Running percentile normalization (5th / 95th percentile) on actor targets |
| **Gradient flow** | Actor receives gradients through critic Q-values (standard SAC) | Straight-through gradients directly through stochastic categorical latent states |
| **Entropy handling** | Fixed alpha = 0.2 on log-prob penalty | Fixed entropy bonus coefficient applied to actor objective |
| **Imagination horizon** | 4 steps | 15 steps (default) |
| **Action rescaling** | Tanh-squashed Gaussian with action-space bounds | Same for continuous; straight-through categorical for discrete |

---

## 4. Encoder / Decoder

| Aspect | car-racing-dreamerv3 | DreamerV3 |
|---|---|---|
| **Depth** | 3 conv layers (3 → 32 → 64 → 128 channels, stride 2) | 4–5 layers, up to 192–512+ channels depending on model size variant |
| **Activation** | ELU | ELU |
| **Output** | Flat 1024-dim → linear projection → embedding | Flat features concatenated with recurrent state `h_t` |
| **Decoder input** | Embedding only | `h_t + z_t` concatenated |
| **Normalization** | LayerNorm on embeddings post-encode | LayerNorm throughout |

---

## 5. Training Loop Structure

| Aspect | car-racing-dreamerv3 | DreamerV3 |
|---|---|---|
| **Phase separation** | World model and AC trained in the same alternating loop | Strict phases: (1) collect data → (2) train WM on sequences → (3) imagine → (4) train AC on imagination |
| **Imagination horizon** | 4 steps | 15 steps |
| **Buffer** | Uniform replay of individual transitions | Uniform replay of **sequences**, with sequence-aware sampling |
| **Target network** | Soft update once per episode (tau = 0.005) | EMA target network, updated every gradient step |

---

## 6. Pros and Cons

### car-racing-dreamerv3

**Pros**

- **Simplicity.** A feedforward world model is far easier to implement, debug, and reason
  about than the RSSM. The entire dynamics model is ~50 lines.
- **SAC is principled for continuous control.** Twin critics, entropy regularization, and the
  reparameterization trick are well-studied and give good sample efficiency guarantees in
  the tabular / low-dimensional setting.
- **Mixed sampling grounds the AC in reality.** Using real replay data gives the critic
  ground-truth reward signals as a regularizer against world model errors. This is especially
  valuable early in training when the world model is still inaccurate.
- **No sequential data requirements.** Training on individual transitions enables simple
  uniform sampling and avoids recurrence implementation bugs.
- **No KL tuning.** Free bits, KL balancing coefficients, and categorical temperature are
  notoriously sensitive; none of these apply here.

**Cons**

- **No temporal memory.** A single frame cannot encode the car's velocity, angular momentum,
  or turn direction. The actor must make continuous control decisions with no motion signal.
  DreamerV3's GRU accumulates this context automatically across the episode.
- **Deterministic latent — no uncertainty modeling.** When the world model is wrong, there
  is no mechanism to express or bound that uncertainty. The AC treats imagined rollouts as
  ground truth.
- **1-step TD creates high bias.** With γ = 0.99, a single bootstrapped step introduces
  significant bias in Q-estimates. λ-returns would reduce this substantially.
- **Dynamics supervision is indirect.** MSE between embeddings assumes the embedding space
  is already well-structured. DreamerV3's KL approach learns representation and dynamics
  jointly without needing a target embedding.
- **Short imagination horizon (4 steps).** Limits how far ahead the AC can plan. At 4 steps
  with γ = 0.99, 96% of the discounted credit comes from these 4 steps. DreamerV3 at 15
  steps can assign credit for decisions that have delayed consequences.

---

### DreamerV3

**Pros**

- **Recurrent state = temporal context.** The GRU carries velocity, momentum, and history
  into every decision — critical for racing, where the consequence of a steering action
  takes several frames to manifest.
- **Stochastic categorical latents enable richer representations.** The prior/posterior
  structure allows the model to represent ambiguous states and learn more structured
  embeddings via the information bottleneck pressure of the KL loss.
- **λ-returns reduce value estimation variance.** Multi-step returns are better for
  environments where credit is delayed or sparse.
- **Symlog makes training scale-agnostic.** The same hyperparameters work across environments
  with rewards ranging from −1 to +10,000 without manual tuning.
- **Pure imagination for AC is architecturally clean.** No real/imagined mixing; the AC
  gradient signal is internally consistent.

**Cons**

- **Requires sequential data.** Cannot train on random transitions; sequences must be stored
  and sampled as chunks. This adds infrastructure complexity and memory overhead.
- **Categorical straight-through gradients are finicky.** The temperature parameter matters;
  too high and the latent is uninformative, too low and gradients vanish.
- **KL free bits and balancing are sensitive hyperparameters.** Wrong values cause posterior
  collapse (encoder ignores observations) or prior collapse (dynamics model stops learning).
- **Pure imagination vulnerability.** If the world model is wrong, the AC has no real-data
  anchor. The current implementation's mixed sampling is a direct mitigation for this risk.
- **Significantly more code.** The RSSM alone is roughly 3–4× the complexity of the current
  dynamics model. Correct sequence handling, posterior inference, and straight-through
  sampling each add implementation surface.

---

## 7. The Single Biggest Gap

**The absence of recurrence.**

In CarRacing-v3, a single frame does not contain enough information to determine the car's
velocity or the direction of a turn. An agent acting on a single embedding must implicitly
infer motion from visual blur and positional change, which is a much harder task than
acting on an explicit velocity signal.

DreamerV3's GRU recurrent state `h_t` accumulates this information across the episode
automatically. Every AC decision is conditioned on the full history of observations and
actions, encoded in `h_t`. The current implementation conditions the actor only on a single
frame embedding, discarding all temporal context between steps.

**Recommended next experiment:** Add a single GRU layer between the encoder and the AC
inputs. The actor and critic would receive `h_t = GRU(embed_t, h_{t-1})` instead of
`embed_t` directly. This does not require the full RSSM — no stochastic states, no KL loss,
no sequence sampling — but it would give the policy temporal context at a fraction of the
implementation cost. The world model dynamics model could simultaneously be updated to
predict `h_{t+1}` from `h_t + action`, preserving the imagination capability.

---

## 8. Summary Table

| Property | car-racing-dreamerv3 | DreamerV3 |
|---|---|---|
| Recurrent state | No | Yes (GRU) |
| Stochastic latent | No | Yes (categorical) |
| KL regularization | No | Yes (with free bits + balancing) |
| Symlog reward transform | No | Yes |
| λ-returns | No (1-step TD) | Yes (λ = 0.95) |
| Value normalization | No | Yes (percentile) |
| AC training data | 75% real + 25% imagined | 100% imagined |
| Imagination horizon | 4 steps | 15 steps |
| Training data shape | Transitions | Sequences |
| Implementation complexity | Low | High |
| Temporal context | None | Full episode history |

---

*References: Hafner et al. (2023). "Mastering Diverse Domains with World Models." arXiv:2301.04104*
