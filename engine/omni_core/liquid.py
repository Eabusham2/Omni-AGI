"""Closed-form continuous and liquid time-constant controllers."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .model import BitLinear


class CfCCell(nn.Module):
    """A compact closed-form continuous-time recurrent cell.

    The learned time constant controls an analytic exponential interpolation,
    so a larger elapsed time advances state farther without an RNN unroll.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        joined = input_size + hidden_size
        self.candidate = BitLinear(joined, hidden_size, bias=True)
        self.gate = BitLinear(joined, hidden_size, bias=True)
        self.time_constant = BitLinear(joined, hidden_size, bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed: float = 1.0,
    ) -> torch.Tensor:
        if state is None:
            state = torch.zeros(
                inputs.shape[0],
                self.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
        joined = torch.cat([inputs, state], dim=-1)
        candidate = torch.tanh(self.candidate(joined))
        gate = torch.sigmoid(self.gate(joined))
        tau = F.softplus(self.time_constant(joined)) + 0.1
        alpha = 1.0 - torch.exp(
            -torch.as_tensor(elapsed, dtype=inputs.dtype, device=inputs.device)
            / tau
        )
        return state + alpha * gate * (candidate - state)


class LTCCell(nn.Module):
    """Euler-solved liquid time-constant cell with bounded conductances."""

    def __init__(self, input_size: int, hidden_size: int, solver_steps: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver_steps = max(1, int(solver_steps))
        joined = input_size + hidden_size
        self.conductance = BitLinear(joined, hidden_size, bias=True)
        self.reversal = BitLinear(joined, hidden_size, bias=True)
        self.input_drive = BitLinear(input_size, hidden_size, bias=True)
        self.leak_logit = nn.Parameter(torch.full((hidden_size,), -0.25))
        self.capacitance_logit = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed: float = 1.0,
    ) -> torch.Tensor:
        if state is None:
            state = torch.zeros(
                inputs.shape[0],
                self.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
        dt = float(elapsed) / float(self.solver_steps)
        leak = F.softplus(self.leak_logit) + 0.05
        capacitance = F.softplus(self.capacitance_logit) + 0.25
        drive = self.input_drive(inputs)
        hidden = state
        for _ in range(self.solver_steps):
            joined = torch.cat([inputs, hidden], dim=-1)
            conductance = torch.sigmoid(self.conductance(joined))
            reversal = torch.tanh(self.reversal(joined))
            derivative = (
                -leak * hidden
                + conductance * (reversal - hidden)
                + drive
            ) / capacitance
            hidden = torch.tanh(hidden + dt * derivative)
        return hidden


class LiquidController(nn.Module):
    """Turns temporal state into interpretable memory/control modulation."""

    def __init__(
        self,
        dimensions: int,
        mode: str = "cfc",
        solver_steps: int = 3,
    ):
        super().__init__()
        if mode == "cfc":
            self.cell = CfCCell(dimensions, dimensions)
        elif mode == "ltc":
            self.cell = LTCCell(dimensions, dimensions, solver_steps=solver_steps)
        else:
            raise ValueError("mode must be cfc or ltc")
        self.mode = mode
        self.controls = BitLinear(dimensions, 4, bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden = self.cell(inputs, state=state, elapsed=elapsed)
        raw = torch.sigmoid(self.controls(hidden))
        controls = {
            "retention": raw[:, 0],
            "threshold_offset": (raw[:, 1] - 0.5) * 0.3,
            "noise_scale": 0.25 + raw[:, 2] * 1.5,
            "ponder_scale": 0.5 + raw[:, 3] * 2.5,
        }
        return hidden, controls
