""" Reward model for Ego-allo. """

import torch
import torch.nn as nn

from egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose

from egoallo.transforms import SE3, SO3

from egoallo.metrics_helpers import (
    compute_foot_contact_reward,
    compute_foot_skate_reward,
    compute_mpjpe_reward,
    compute_groundpenetrate_reward,
    compute_mpjve_reward,
    jitter_reward
)


device = torch.device("cuda")

class RewardModel(nn.Module):
    def __init__(self, encoder: int):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
    
    
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



def compute_rewards(samples, batch, body_model, reward_model = None) -> torch.Tensor:
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
        metric_coefficient = 1000.0
    )
    
    pampjpe_reward = compute_mpjpe_reward(
        label_T_world_root=label_joints.T_world_root,
        label_Ts_world_joint=label_joints.Ts_world_joint[:, :, :21, :],
        pred_T_world_root=pred_joints.T_world_root,
        pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :],
        per_frame_procrustes_align=True,
        metric_coefficient = 1000.0
    )
        
    mpjve_reward = compute_mpjve_reward(
        label_T_world_root=label_joints.T_world_root,
        label_Ts_world_joint=label_joints.Ts_world_joint[:, :, :21, :],
        pred_T_world_root=pred_joints.T_world_root,
        pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :],
        metric_coefficient = 1000.0
    )
    
    gp_reward = compute_groundpenetrate_reward(
        label_T_world_root=label_joints.T_world_root,
        label_Ts_world_joint=label_joints.Ts_world_joint[:, :, :21, :],
        pred_T_world_root=pred_joints.T_world_root,
        pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :],
        metric_coefficient = 1000.0
    )
    
    #foot_contact_reward = compute_foot_contact_reward(pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :], return_tensor=True)
    
    # pred_jitter = jitter_reward(
    #     label_T_world_root=label_joints.T_world_root,
    #     label_Ts_world_joint=label_joints.Ts_world_joint[:, :, :21, :],
    #     pred_T_world_root=pred_joints.T_world_root,
    #     pred_Ts_world_joint=pred_joints.Ts_world_joint[:, :, :21, :],
    # )

    if reward_model is not None:
        # prepare input for reward model
        # Convert 7D rotation to 6D rotation representation
        pred_joints_root_quaternion = pred_joints.T_world_root[:, :, None, :4]
        pred_joints_body_quaternion = pred_joints.Ts_world_joint[:, :, :21, :4]
        pred_joints_lhand_quaternion = pred_joints.Ts_world_joint[:, :, [25], :4]
        pred_joints_rhand_quaternion = pred_joints.Ts_world_joint[:, :, [40], :4]
        pred_joints_quaternion = torch.cat([
            pred_joints_root_quaternion,
            pred_joints_body_quaternion,
            pred_joints_lhand_quaternion,
            pred_joints_rhand_quaternion
        ], dim = 2)
        
        pred_joints_axis_angle = quaternion_to_axis_angle(pred_joints_quaternion)
        root_xyz = pred_joints.T_world_root[:, :, None, 4:]
        critic_input = torch.cat([pred_joints_axis_angle, root_xyz], dim=-2)
        critic_score = reward_model.module.batch_critic(critic_input).reshape(-1)
        reward_dict = {
            "foot_skate_reward": foot_skate_reward,
            "mpjpe_reward": mpjpe_reward,
            "pampjpe_reward": pampjpe_reward,
            #"pred_jitter": pred_jitter,
            "gp_reward": gp_reward,
            "mpjve_reward": mpjve_reward,
            #"foot_contact_reward": foot_contact_reward,
            "critic_score": critic_score
        } 
        reward = - (foot_skate_reward + mpjpe_reward + pampjpe_reward + gp_reward + mpjve_reward) + critic_score * 10 #+ foot_contact_reward 
    else:
        reward_dict = {
            "foot_skate_reward": foot_skate_reward,
            "mpjpe_reward": mpjpe_reward,
            "pampjpe_reward": pampjpe_reward,
            #"pred_jitter": pred_jitter,
            "gp_reward": gp_reward,
            "mpjve_reward": mpjve_reward,
            #"foot_contact_reward": foot_contact_reward
        } 
        reward = - (foot_skate_reward + mpjpe_reward + pampjpe_reward + gp_reward + mpjve_reward) # + foot_contact_reward 
    
    return reward, reward_dict