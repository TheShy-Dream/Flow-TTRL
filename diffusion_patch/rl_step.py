import torch
from collections import defaultdict


class RLSampler:
    def __init__(self):
        self.samples = []
        self.id_to_index = {} 

    def rollout(self, sample_id, timesteps, latents, next_latents, log_probs,prev_latents_mean, std_dev_t,rewards, prompt_ids=None):

        if sample_id in self.id_to_index:
            raise ValueError(f"ID {sample_id} already exists!")

        item = {
            "id": sample_id,
            "timesteps": timesteps,
            "latents": latents,
            "next_latents": next_latents,
            "log_probs": log_probs,
            "prev_latents_mean": prev_latents_mean,
            "std_dev_t": std_dev_t,
            "rewards": rewards,
        }
        if prompt_ids is not None:
            item["prompt_ids"] = prompt_ids

        self.samples.append(item)
        self.id_to_index[sample_id] = len(self.samples) - 1

    def update_reward(self, sample_id, new_rewards):
        if sample_id not in self.id_to_index:
            raise ValueError(f"{sample_id} does not exists!")

        index = self.id_to_index[sample_id]
        self.samples[index]["rewards"] = new_rewards

    def get_sample(self, sample_id):
        if sample_id not in self.id_to_index:
            raise ValueError(f"{sample_id} does not exists!")

        index = self.id_to_index[sample_id]
        return self.samples[index]

    def clear(self):
        self.samples = []
        self.id_to_index = {}

    def __len__(self):
        return len(self.samples)

    def size(self):
        return len(self.samples)

    def get_batch(self):
        if not self.samples:
            return None
        batch = defaultdict(list)
        for s in self.samples:
            for k, v in s.items():
                if k == "id":
                    continue
                batch[k].append(v)
        for k in batch:
            batch[k] = torch.cat(batch[k], dim=0)
        return batch

    def compute_and_update_advantages(self, by_prompt=False):

        if len(self.samples) == 0:
            return

        avg_rewards = []
        prompt_keys = []

        for s in self.samples:
            r = s["rewards"]
            avg_rewards.append(r.mean())

            if by_prompt and "prompt_ids" in s:
                prompt_keys.append(tuple(s["prompt_ids"].tolist()))
            else:
                prompt_keys.append(None)

        avg_rewards = torch.stack(avg_rewards)  # (N,)

        advantages = torch.zeros_like(avg_rewards)

        if by_prompt:
            group_map = defaultdict(list)
            for idx, key in enumerate(prompt_keys):
                group_map[key].append(idx)

            for _, idx_list in group_map.items():
                group_values = avg_rewards[idx_list]
                mean = group_values.mean()
                std = group_values.std() + 1e-4
                advantages[idx_list] = (group_values - mean) / std

        else:
            mean = avg_rewards.mean()
            std = avg_rewards.std() + 1e-4
            advantages = (avg_rewards - mean) / std

        for s, adv in zip(self.samples, advantages):
            s["advantages"] = adv.unsqueeze(0)  

        return advantages

    def get_all_ids(self):
        return list(self.id_to_index.keys())

    def remove_sample(self, sample_id):
        if sample_id not in self.id_to_index:
            raise ValueError(f"ID {sample_id} does not exists!")

        index = self.id_to_index[sample_id]

        del self.samples[index]

        del self.id_to_index[sample_id]

        for sid, idx in self.id_to_index.items():
            if idx > index:
                self.id_to_index[sid] = idx - 1

    def has_sample(self, sample_id):
        return sample_id in self.id_to_index

    def get_all_key_tensors(self, key):

        if not self.samples:
            return None

        if key not in self.samples[0]:
            raise ValueError(f"'{key}' does not exists!")

        tensors = [s[key].unsqueeze(0) if s[key].dim() == 0 else s[key] for s in self.samples]
        return torch.cat(tensors, dim=0)