import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, img_size=64, proprio_dim=50, embed_dim=1024):
        super(Backbone, self).__init__()
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

        # Proprioceptive state processing network (MLP)
        self.proprio_mlp = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
        )

        # Calculate the size of the flattened CNN outputs
        # A dummy forward pass helps in determining the flat feature size
        with torch.inference_mode():
            dummy_img = torch.zeros(1, 3, img_size, img_size)
            dummy_depth = torch.zeros(1, 1, img_size, img_size)
            img_feature_size = self.image_conv(dummy_img).shape[1]
            depth_feature_size = self.depth_conv(dummy_depth).shape[1]

        # Fusion layer
        fused_size = img_feature_size + depth_feature_size + embed_dim // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_size, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, obs):

        img = (obs["robot0_eye_in_hand_image"] - 128.0) / 128.0  # Normalize to [-1, 1]
        img = img.permute(
            0, 3, 1, 2
        ).contiguous()  # Change to (batch, channel, height, width)

        depth = (obs["robot0_eye_in_hand_depth"] - 0.5) / 0.5  # Normalize to [-1, 1]
        depth = depth.permute(0, 3, 1, 2).contiguous()

        proprio = obs["robot0_proprio-state"]

        img_features = self.image_conv(img)
        depth_features = self.depth_conv(depth)
        proprio_features = self.proprio_mlp(proprio)

        # Concatenate features
        fused_features = torch.cat(
            (img_features, depth_features, proprio_features), dim=1
        )
        fused_output = self.fusion_layer(fused_features)

        return fused_output


class ActorCritic(nn.Module):
    def __init__(
        self,
        action_dim,
        img_size=64,
        proprio_dim=50,
        fixed_policy_variance=True,
        action_std=0.5,
        embed_dim=1024,
        object_dim=10,
        predict_object_state=False,
    ):
        super(ActorCritic, self).__init__()
        self.fixed_policy_variance = fixed_policy_variance
        self.predict_object_state = predict_object_state

        self.backbone = Backbone(img_size, proprio_dim, embed_dim)

        # Actor head
        if fixed_policy_variance:
            action_var = torch.full((action_dim,), action_std**2)
            self.register_buffer("action_var", action_var)
            self.actor = nn.Linear(embed_dim, action_dim)
        else:
            self.actor = nn.Linear(
                embed_dim, action_dim * 2
            )  # Output both mean and log variance

        # Critic head
        self.critic = nn.Linear(embed_dim, 1)

        if predict_object_state:
            self.object_predictor = nn.Linear(embed_dim, object_dim)

    def forward(self, obs):

        fused_output = self.backbone(obs)

        action_mean = self.actor(fused_output)

        if self.fixed_policy_variance:
            action_var = self.action_var.expand_as(action_mean)
        else:
            action_mean, action_logvar = action_mean.chunk(2, dim=-1)
            action_logvar = torch.clamp(
                action_logvar, -5, -0.5
            )  # Clamp log variance for stability
            action_var = torch.exp(action_logvar)

        state_value = self.critic(fused_output)

        if self.predict_object_state:
            object_state_pred = self.object_predictor(fused_output)
        else:
            object_state_pred = None

        return action_mean, action_var, state_value, object_state_pred
