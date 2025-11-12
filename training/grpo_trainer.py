import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import trange
import random

# ============================================================
#   GRPO Trainer 模块（通用化版本）
# ============================================================

class GRPOTrainer:
    """
    Generalized GRPO (Generalized Reinforcement Policy Optimization) trainer.
    Implements PPO-style clipped loss + multi-term reward:
        total_reward = W_CONS * consistency + W_FIN * finbert_align + W_CAL * calibration
    """

    def __init__(
        self,
        student_model,
        student_probs_fn,
        pick_one_pert_fn,
        total_reward_fn,
        label_order,
        to_model_device_fn,
        lr: float = 1e-5,
        eps_clip: float = 0.1,
        kl_coef: float = 0.01,
        ema_alpha: float = 0.9,
        grad_clip: float = 1.0,
    ):
        """
        Args:
            student_model: StudentClassifierForCausalLM 包装器实例
            student_probs_fn: 无梯度预测函数 (texts -> [B,3] 概率)
            pick_one_pert_fn: 函数，返回扰动样本 pick_one_pert(text)
            total_reward_fn: 函数，计算总奖励 total_reward(p_x, p_xt, texts)
            label_order: 标签顺序 ["negative", "neutral", "positive"]
            to_model_device_fn: helper 将 inputs.to(model_device)
            lr, eps_clip, kl_coef, ema_alpha, grad_clip: 超参数
        """
        self.student_model = student_model
        self.student_probs_fn = student_probs_fn
        self.pick_one_pert_fn = pick_one_pert_fn
        self.total_reward_fn = total_reward_fn
        self.label_order = label_order
        self.to_model_device_fn = to_model_device_fn

        self.eps_clip = eps_clip
        self.kl_coef = kl_coef
        self.ema_alpha = ema_alpha
        self.grad_clip = grad_clip
        self.baseline_ema = 0.0

        # 仅优化 LoRA 或可训练参数
        self.trainable_params = [
            p for p in student_model.causal_lm.parameters() if p.requires_grad
        ]
        print(f"[GRPOTrainer] Training {len(self.trainable_params)} parameters.")
        self.optimizer = AdamW(self.trainable_params, lr=lr)

    # ----------------------------------------------------------
    # 无梯度旧策略
    # ----------------------------------------------------------
    @torch.no_grad()
    def _batched_probs_old(self, texts):
        self.student_model.causal_lm.eval()
        p = self.student_probs_fn(texts)  # [B,3]
        actions = p.argmax(dim=-1)
        logp = (
            p.clamp_min(1e-12)
            .log()
            .gather(1, actions[:, None])
            .squeeze(1)
        )
        return p, actions, logp

    # ----------------------------------------------------------
    # 带梯度新策略
    # ----------------------------------------------------------
    def _batched_probs_with_grad(self, texts):
        self.student_model.causal_lm.train()
        prompts = self.student_model._build_prompts(texts)
        toks = self.student_model.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.student_model.max_length,
        )
        toks = self.to_model_device_fn(toks, self.student_model.causal_lm)
        out = self.student_model.causal_lm(**toks)
        last_idx = toks["attention_mask"].sum(dim=1) - 1
        logits_last = out.logits[torch.arange(out.logits.size(0)), last_idx, :]

        label_scores = []
        for label in self.label_order:
            ids = self.student_model.label_token_ids[label]
            if len(ids) == 1:
                score = logits_last[:, ids[0]]
            else:
                score = torch.logsumexp(logits_last[:, ids], dim=1)
            label_scores.append(score)
        scores = torch.stack(label_scores, dim=1)
        probs = scores.softmax(dim=1)
        actions = probs.argmax(dim=-1)
        logp = (
            probs.clamp_min(1e-12)
            .log()
            .gather(1, actions[:, None])
            .squeeze(1)
        )
        return probs, actions, logp

    # ----------------------------------------------------------
    # 单步训练
    # ----------------------------------------------------------
    def grpo_train_step(self, batch_texts):
        texts, texts_tilde = [], []
        for t in batch_texts:
            pt = self.pick_one_pert_fn(t)
            if pt is not None:
                texts.append(t)
                texts_tilde.append(pt)
        if not texts:
            return {"skipped": True}

        # 旧策略
        p_old, _, logp_old = self._batched_probs_old(texts)
        # 新策略
        p_new, _, logp_new = self._batched_probs_with_grad(texts)
        p_xt, _, _ = self._batched_probs_with_grad(texts_tilde)

        # 奖励 + 优势
        with torch.no_grad():
            R = self.total_reward_fn(p_new.detach(), p_xt.detach(), texts)
            self.baseline_ema = (
                self.ema_alpha * self.baseline_ema + (1 - self.ema_alpha) * R.mean().item()
            )
            adv = R - self.baseline_ema

        # PPO clipped loss
        ratio = torch.exp(logp_new - logp_old)
        unclipped = -ratio * adv
        clipped = -torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
        loss_policy = torch.maximum(unclipped, clipped).mean()

        # KL regularization
        kl_reg = F.kl_div(
            p_old.clamp_min(1e-12).log(),
            p_new.clamp_min(1e-12),
            reduction="batchmean",
        )
        loss = loss_policy + self.kl_coef * kl_reg

        # 优化
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            flip_rate = (p_new.argmax(-1) != p_xt.argmax(-1)).float().mean().item()

        return {
            "loss": float(loss.item()),
            "R": float(R.mean().item()),
            "flip": flip_rate,
        }

    # ----------------------------------------------------------
    # 主训练循环
    # ----------------------------------------------------------
    def run(self, train_texts, batch_size=32, steps=200):
        for step in trange(steps, desc="[GRPO Training]"):
            batch = random.sample(train_texts, k=min(batch_size, len(train_texts)))
            stats = self.grpo_train_step(batch)
            if stats.get("skipped"):
                continue
            if step % 10 == 0:
                print(
                    f"[{step}] loss={stats['loss']:.4f}  "
                    f"R={stats['R']:.3f}  flip={stats['flip']:.3f}"
                )
