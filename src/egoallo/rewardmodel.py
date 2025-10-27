""" Reward model for Ego-allo. """

import torch
import torch.nn as nn

from egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose

from egoallo.transforms import SE3, SO3

from egoallo.metrics_helpers import (
    compute_foot_contact,
    compute_foot_skate,
    compute_head_trans,
    compute_mpjpe_reward,
)

device = torch.device("cuda")

class RewardModel(nn.Module):
    def __init__(self, encoder: int):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
    


def get_joints(samples, batch, body_model):
    
    # pred result
    pred_posed = body_model.with_shape(samples[-1].betas).with_pose(
        T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
        local_quats=SO3.from_matrix(
            torch.cat([samples[-1].body_rotmats, samples[-1].hand_rotmats], dim=2)
        ).wxyz,
    )
    pred_posed = pred_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(pred_posed, batch.T_world_cpf[:, 1:, ...])
    )
    
    # gt
    label_posed = body_model.with_shape(batch.betas[:, 1:, ...]).with_pose(
        batch.T_world_root[:, 1:, ...],
        torch.cat(
            [
                batch.body_quats[:, 1:, ...],
                batch.hand_quats[:, 1:, ...],
            ],
            dim=2,
        ),
    )

    return pred_posed, label_posed



def compute_rewards(samples: torch.Tensor, batch, body_model) -> torch.Tensor:
    """ Compute rewards for the given samples and batch data.

    Args:
        samples: Tensor of shape (B, T, D) representing the generated motion samples.
        batch: Dictionary containing the batch data.
        body_model: The body model used for kinematic computations.
    """
    # Obtain the matrics
    pred_joints, label_joints = get_joints(samples, batch, body_model)
    
    # get the reward
    batch_size = samples[0].betas.shape[0]
    
    foot_skate_reward = compute_foot_skate_reward(pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :], return_tensor=True)
    
    mpjpe_reward = compute_mpjpe_reward(
        label_T_world_root=label_joints.T_world_root,
        label_Ts_world_joint=label_joints.Ts_world_joint[:, :, :21, :],
        pred_T_world_root=pred_joints.T_world_root,
        pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :],
        per_frame_procrustes_align=False,
    )
    
    pampjpe_reward = compute_mpjpe_reward(
        label_T_world_root=label_joints.T_world_root,
        label_Ts_world_joint=label_joints.Ts_world_joint[:, :, :21, :],
        pred_T_world_root=pred_joints.T_world_root,
        pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :],
        per_frame_procrustes_align=True,
    )
    
    reward = - (mpjpe_reward + pampjpe_reward)
    
    return reward # foot_skate_reward