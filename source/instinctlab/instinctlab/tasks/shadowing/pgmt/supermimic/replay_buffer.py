import torch
from dataclasses import dataclass
from tensordict import TensorDict


@dataclass(frozen=True)
class FieldSpec:
    dim: int | tuple[int, ...] = 0  # 0 = scalar (no trailing axis); >0 = vector with trailing dim
    dtype: torch.dtype = torch.float32

    def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
        shape = self.dim if isinstance(self.dim, tuple) else ((self.dim,) if self.dim else ())
        return (num_envs, buffer_size, *shape)


class ReplayBuffer:
    """基于 TensorDict 的多环境循环缓冲区，支持按环境分层采样连续序列。

    存储的批次维度为 [num_envs, buffer_size]，分别对应环境和时间槽。
    每次插入所有环境的一步数据；写满后循环覆盖最旧的数据。

    sample() 先均匀选择有合法序列的环境，再均匀选择该环境的合法起点，
    返回不跨 episode 的完整序列，不进行填充。
    sample_causal() 先均匀选择环境和当前帧，再向前读取历史；
    历史不足或遇到 episode 边界时重复最早可用帧，并返回填充掩码。

    terminated 和 truncated 均视为 episode 结束标记。
    带结束标记的帧可以作为窗口末帧，但不能位于连续窗口内部。
    """

    def __init__(
        self,
        schema: dict[str, FieldSpec],
        buffer_size: int,
        num_envs: int,
        device: str = "cpu",
    ):
        if not {"terminated", "truncated"} <= schema.keys():
            raise ValueError("schema needs 'terminated' and 'truncated'")
        # 每个环境已存储的有效时间步数，不是所有环境的样本总数。
        self.num_transitions = 0
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        # 下一次写入的物理槽位；写满后也指向最旧的数据。
        self._step = 0
        # 各字段共享 [环境, 时间] 批次维度，特征维度由 schema 指定。
        self._storage = TensorDict(
            {
                name: torch.zeros(spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=device)
                for name, spec in schema.items()
            },
            batch_size=[num_envs, buffer_size],
        )

    def insert(self, td: TensorDict) -> None:
        """插入所有环境当前一步的数据，并推进循环写入位置。"""
        if td.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}")
        if set(td.keys()) != set(self._storage.keys()):
            raise ValueError("schema mismatch")
        self._storage[:, self._step] = td
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

    def sample_causal(self, sequence_length: int, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """采样以随机当前帧结尾的 L 帧历史窗口，最新帧位于最后。

        均匀选择环境及其中已存储的当前帧，当前帧的分布不随 L 改变。
        向前保留同一 episode 内的真实历史；超出存储范围或跨越 episode
        边界的位置，用窗口内最早可用帧重复填充，与仿真历史缓存重置时一致。
        之前带结束标记的帧属于上一 episode，不纳入当前帧的历史；
        当前帧本身即使带结束标记也始终保留。

        返回 (window, padded)：
          window：批次维度为 (B, L) 的 TensorDict。
          padded：形状为 (B, L) 的布尔掩码，True 表示重复填充位置。
            调用方可给最早真实帧添加一次噪声，并让填充位置共享该噪声。
        """
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
        if n < 1:
            raise RuntimeError("sample_causal on empty buffer")
        # base 为最旧有效帧的物理槽位。
        base = self._step if n == T else 0
        # 独立、均匀选择环境和当前帧，允许重复采样。
        env_idx = torch.randint(E, (B,), device=dev)
        current_age = torch.randint(n, (B,), device=dev)
        # 构造从当前帧向前延伸的逻辑索引；负数表示超出已有历史。
        ages = current_age[:, None] + (torch.arange(L, device=dev) - (L - 1))  # (B, L)，从旧到新排列
        ended = (self._storage["terminated"] | self._storage["truncated"])[
            env_idx[:, None], (base + ages.clamp(min=0)) % T
        ]
        # 历史帧必须已存储且没有结束标记；当前帧始终有效。
        # 更早位置上的结束标记代表上一 episode 在该帧结束。
        valid = (ages >= 0) & ~ended.bool()
        valid[:, -1] = True
        # 从最新帧向前累计检查，遇到无效位置后，该位置及更早位置均不可达。
        reachable = torch.cummin(valid.flip(-1).int(), dim=-1).values.flip(-1).bool()
        # 找到窗口内最早可达帧；用当前帧索引替代不可达项以计算最小值。
        first_age = torch.where(reachable, ages, current_age[:, None]).amin(dim=-1, keepdim=True)
        # 不可达位置重复最早可达帧，再映射回循环存储的物理槽位。
        slot = (base + torch.where(reachable, ages, first_age)) % T
        return self._storage[env_idx[:, None], slot], ~reachable

    def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
        """按环境分层采样完整的 L 帧序列，不填充、不跨 episode。

        先在有合法窗口的环境中均匀采样，再在该环境的合法起点中均匀采样。
        窗口最后一帧允许带结束标记；返回批次维度为 (B, L) 的 TensorDict。
        """
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
        # 逻辑时间索引 0 为最旧有效帧，n - 1 为最新有效帧。
        # 物理槽位 = (base + age) % T，从而统一处理循环覆盖和未写满情况。
        base = self._step if n == T else 0
        ages = torch.arange(T, device=dev)
        ended = (self._storage["terminated"] | self._storage["truncated"])[:, (base + ages) % T].long()
        # 用前缀和判断窗口内部 [a, a+L-2] 是否无结束标记：
        # c[a+L-1] == c[a]；最后一帧允许结束。尾部补零仅用于切片对齐。
        c = torch.cat([ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1)
        # 同时要求窗口全部位于已存储范围内，排除尾部越界窗口。
        valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
        # 累计合法起点数量，total 为各环境的合法窗口总数。
        csum = valid.cumsum(dim=1, dtype=torch.long)
        total = csum[:, -1]
        if not (total > 0).any():
            raise RuntimeError(f"no valid sequences L={L} n={n}")
        # 有合法窗口的环境权重相同，不按各环境的窗口数量加权。
        env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
        # 在选中环境内随机选第 u 个合法窗口，编号从 0 开始。
        u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
        # 累计数量首次大于 u 的位置，就是该合法窗口的起点。
        age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
        # 展开连续逻辑时间索引，再映射回实际存储槽位。
        storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
        return self._storage[env_idx[:, None], storage_idx]
