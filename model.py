import torch
import torch.nn as nn


# --- 2. Multi-Modal Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, action_dim, img_size=64, proprio_dim=50,
                 fixed_policy_variance=True):
        super(ActorCritic, self).__init__()
        self.fixed_policy_variance = fixed_policy_variance
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
        if fixed_policy_variance:
            action_var = torch.full((action_dim,), 0.5**2, device=self.device)
            self.register_buffer("action_var", action_var)
            self.actor =nn.Linear(512, action_dim)
        else:
            self.actor = nn.Linear(512, action_dim * 2)  # Output both mean and log variance

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
        if self.fixed_policy_variance:
            action_var = self.action_var.expand_as(action_mean)
        else:
            action_mean, action_logvar = action_mean.chunk(2, dim=-1)
            action_logvar = torch.clamp(action_logvar, -20, 2)  # Clamp log variance for stability
            action_var = torch.exp(action_logvar)
        state_value = self.critic(fused_output)

        return action_mean, action_var, state_value


if __name__ == "__main__":
    x = torch.randn(5, 8)  # Batch of 2 RGB images
    x,y = x.chunk(2, dim=-1)
    print(x.shape, y.shape)