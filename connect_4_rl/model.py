import torch.nn as nn
import torch.nn.functional as F


class Connect4Net(nn.Module):
    """
    Neural Network for Connect-4 AlphaZero agent.
    Input: (B, 3, 8, 8) - [Current Player, Opponent, Valid Moves/Target]
    Output:
        - Policy: (B, 8) - Logits for each column
        - Value: (B, 1) - Expected win probability from current state (-1 to 1)
    """
    def __init__(self, num_channels=128, num_res_blocks=10):
        super().__init__()

        # Initial Conv Block
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),  # Reduce channels to 2
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 8)  # Output for 8 columns
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_input(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
