import torch
import torch.nn as nn


# --- 2. Multi-Modal Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, action_dim, img_size=64, proprio_dim=50):
        super(ActorCritic, self).__init__()

        # Image processing network (CNN) for RGB
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Depth processing network (CNN) for Depth
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the size of the flattened CNN outputs
        # A dummy forward pass helps in determining the flat feature size
        with torch.inference_mode():
            dummy_img = torch.zeros(1, 3, img_size, img_size)
            dummy_depth = torch.zeros(1, 1, img_size, img_size)
            img_feature_size = self.image_conv(dummy_img).shape[1]
            depth_feature_size = self.depth_conv(dummy_depth).shape[1]

        # Proprioceptive state processing network (MLP)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Fusion layer
        fused_size = img_feature_size + depth_feature_size + 64
        self.fusion_layer = nn.Sequential(nn.Linear(fused_size, 512), nn.ReLU())

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(512, action_dim),
            nn.Tanh(),  # To scale actions to [-1, 1]
        )

        # Critic head
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        img = obs["agentview_image"] / 255.0  # Normalize pixel values
        img = img.permute(0, 3, 1, 2).contiguous()  # Change to (batch, channel, height, width)
        depth = obs["agentview_depth"]
        depth = torch.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0)
        depth = torch.clamp(depth, 0.0, 1.0)
        depth = depth.permute(0, 3, 1, 2).contiguous()

        proprio = obs["robot0_proprio-state"]
        proprio = (proprio - proprio.mean(dim=1, keepdim=True)
                   ) / (proprio.std(dim=1, keepdim=True) + 1e-8
                        ).contiguous()

        # Permute image dimensions to be (batch, channel, height, width)
        
        

        img_features = self.image_conv(img)
        depth_features = self.depth_conv(depth)
        proprio_features = self.proprio_mlp(proprio)

        # Concatenate features
        fused_features = torch.cat(
            (img_features, depth_features, proprio_features), dim=1
        )
        fused_output = self.fusion_layer(fused_features)

        action_mean = self.actor(fused_output)
        state_value = self.critic(fused_output)

        return action_mean, state_value
