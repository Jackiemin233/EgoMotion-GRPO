"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

"""
CUDA_VISIBLE_DEVICES=1 python --config.experiment-name debug_reward --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=4,5,6,7 1c_train_motion_prior_GRPO.py accelerate launch  --config.experiment-name debug_reward --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=4 1c_train_motion_prior_GRPO.py accelerate launch  --config.experiment-name ours --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=4 python 1c_train_motion_prior_GRPO.py --config.experiment-name ours --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=1 python 1c_train_motion_prior_GRPO.py --config.experiment-name debug_reward_noperceptual --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=6 python 1c_train_motion_prior_GRPO.py --config.experiment-name debug_reward_noperceptual --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=7 python 1c_train_motion_prior_GRPO.py --config.experiment-name debug_reward_withperceptual --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=3 python 1c_train_motion_prior_GRPO.py --config.experiment-name debug_reward_withoutgp --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=0 python 1c_train_motion_prior_GRPO.py --config.experiment-name debug_reward_lr5e2 --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=0 python 1c_train_motion_prior_GRPO.py --config.experiment-name debug_simplereward --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=1 python 1c_train_motion_prior_GRPO.py --config.experiment-name debug_simplereward_justeva --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt




CUDA_VISIBLE_DEVICES=2 python 1c_train_motion_prior_grpo.py --config.experiment-name debug_mpjpe_100 --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=7 python 1c_train_motion_prior_grpo.py --config.experiment-name debug_mpjpe_pampjpe_100 --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=7 python 1c_train_motion_prior_grpo.py --config.experiment-name baseline --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

CUDA_VISIBLE_DEVICES=7 python 1c_train_motion_prior_grpo.py --config.experiment-name baseline_wofs --config.dataset-hdf5-path ./data/egoalgo_no_skating_dataset.hdf5 --config.dataset-files-path ./data/egoalgo_no_skating_dataset_files.txt

tensorboard --logdir ./
"""

import dataclasses
import shutil
from pathlib import Path
from typing import Literal

import tensorboardX
import torch.optim.lr_scheduler
import torch.utils.data
import tyro
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, set_seed
from loguru import logger

from egoallo import network, training_loss, training_utils
from egoallo.data.amass import EgoAmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass

from egoallo.sampling import run_sampling_with_logprob
from egoallo.inference_utils import load_denoiser

from egoallo import fncsmpl

from egoallo.rewardmodel import compute_rewards
from egoallo.sampling import CosineNoiseScheduleConstants, quadratic_ts

from egoallo.vis_helpers import vis_meshes

from tqdm import tqdm
import os

# for reward model
from thirdparty.MotionCritic.MotionCritic.lib.model.load_critic import load_critic

@dataclasses.dataclass(frozen=True)
class EgoAlloTrainConfig:
    experiment_name: str
    dataset_hdf5_path: Path
    dataset_files_path: Path
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    model: network.EgoDenoiserConfig = network.EgoDenoiserConfig()
    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()
    
    # GRPO Group Size
    group_size: int = 16
    num_inner_epochs: int = 1
    eta : float = 0.8
    
    # Dataset arguments.
    batch_size: int = 64
    """Effective batch size."""
    num_workers: int = 4
    subseq_len: int = 128
    dataset_slice_strategy: Literal[
        "deterministic", "random_uniform_len", "random_variable_len"
    ] = "random_uniform_len"
    dataset_slice_random_variable_len_proportion: float = 0.3
    """Only used if dataset_slice_strategy == 'random_variable_len'."""
    train_splits: tuple[Literal["train", "val", "test", "just_humaneva"], ...] = (
        "train",
        "val",
    )
    gradient_accumulation_steps = 1 # 8

    # Optimizer options.
    learning_rate: float = 1e-5 #1e-5 #1e-5 #5e-2
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-4
    adam_epsilon = 1e-8
    
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Reward model
    enable_reward_model: bool = True
    reward_model_path: str = "./data/motioncritic_pre.pth"
    
    pose_weight: float = 1.0 #100.0
    velocity_weight: float = 0.2
    rotation_weight: float = 1.0
    fs_weight: float = 0#1.0
    critic_weight: float = 0.2
    
    # training related 
    save_ckpt_freq = 100
    save_vis_freq = 5
    seed: int = 42

    # visualization related
    vis_interval = 20
    
def get_experiment_dir(experiment_name: str, version: int = 0) -> Path:
    """Creates a directory to put experiment files in, suffixed with a version
    number. Similar to PyTorch lightning."""
    experiment_dir = (
        Path(__file__).absolute().parent
        / "experiments"
        / experiment_name
        / f"v{version}"
    )
    if experiment_dir.exists():
        return get_experiment_dir(experiment_name, version + 1)
    else:
        return experiment_dir


def run_training(
    config: EgoAlloTrainConfig,
    restore_checkpoint_dir: Path | None = None,
) -> None:
    # Set up experiment directory + HF accelerate.
    # We're getting to manage logging, checkpoint directories, etc manually,
    # and just use `accelerate` for distibuted training.
    
    experiment_dir = get_experiment_dir(config.experiment_name)
    assert not experiment_dir.exists()
    
    num_train_timesteps = 30 - 1
    
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        dataloader_config=DataLoaderConfiguration(split_batches=True),
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.gradient_accumulation_steps * num_train_timesteps,
    )
    
    set_seed(config.seed, device_specific=True)
    
    writer = (
        tensorboardX.SummaryWriter(logdir=str(experiment_dir), flush_secs=10)
        if accelerator.is_main_process
        else None
    )
    device = accelerator.device

    # Initialize experiment.
    if accelerator.is_main_process:
        training_utils.pdb_safety_net()

        # Save various things that might be useful.
        experiment_dir.mkdir(exist_ok=True, parents=True)
        (experiment_dir / "git_commit.txt").write_text(
            training_utils.get_git_commit_hash()
        )
        (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        (experiment_dir / "run_config.yaml").write_text(yaml.dump(config))
        (experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))
        # Visualization path
        (os.makedirs(f'{str(experiment_dir)}/visualization', exist_ok=True))

        # Add hyperparameters to TensorBoard.
        assert writer is not None
        writer.add_hparams(
            hparam_dict=training_utils.flattened_hparam_dict_from_dataclass(config),
            metric_dict={},
            name=".",  # Hack to avoid timestamped subdirectory.
        )

        # Write logs to file.
        logger.add(experiment_dir / "trainlog.log", rotation="100 MB")
        
        # Train!
        samples_per_epoch = (
            config.batch_size
            * accelerator.num_processes
            * config.group_size
        )
        total_train_batch_size = (
            config.batch_size
            * accelerator.num_processes
            * config.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Sample batch size per device = {config.batch_size}")
        logger.info(f"  Train batch size per device = {config.batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}"
        )
        logger.info("")
        logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {config.num_inner_epochs}")
        

    # Setup.
    model = load_denoiser(config.checkpoint_dir).to(device)
    model.latent_from_cond.eval()
    model.latent_from_cond.requires_grad_(False)  # Freeze conditional encoder.
    
    logger.info("Loaded pretrained model from {}".format(config.checkpoint_dir))
    body_model = fncsmpl.SmplhModel.load(config.smplh_npz_path).to(device)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=EgoAmassHdf5Dataset(
            config.dataset_hdf5_path,
            config.dataset_files_path,
            splits= config.train_splits, # ("test",),("just_TotalCapture",), #
            subseq_len=config.subseq_len,
            cache_files=True,
            slice_strategy=config.dataset_slice_strategy,
            random_variable_len_proportion=config.dataset_slice_random_variable_len_proportion,
        ),
        batch_size=config.batch_size,
        shuffle = True, # TODO: Should be true for training
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        collate_fn=collate_dataclass,
        drop_last=True,
    )
    
    optim = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # import reward model
    if config.enable_reward_model and config.reward_model_path is not None:
        reward_model = load_critic("./data/motioncritic_pre.pth", device)
        reward_model.eval()
        reward_model.requires_grad_(False)
        logger.info("Loaded reward model from {}".format(config.reward_model_path))
    else:
        reward_model = None
        logger.info("No reward model loaded.")
    
    # HF accelerate setup. We use this for parallelism, etc!
    model, train_loader, optim = accelerator.prepare(
        model, train_loader, optim
    )
    if reward_model is not None:
        reward_model = accelerator.prepare(reward_model)

    # Restore an existing model checkpoint.
    if restore_checkpoint_dir is not None:
        accelerator.load_state(str(restore_checkpoint_dir))

    # Get the initial step count.
    if restore_checkpoint_dir is not None and restore_checkpoint_dir.name.startswith(
        "checkpoint_"
    ):
        epoch = int(restore_checkpoint_dir.name.partition("_")[2])
    else:
        epoch = 0
        assert epoch == 0 or restore_checkpoint_dir is not None, epoch

    # Save an initial checkpoint. Not a big deal but currently this has an
    # off-by-one error, in that `step` means something different in this
    # checkpoint vs the others.
    accelerator.save_state(str(experiment_dir / f"checkpoints_{epoch}"))

    # Run training loop!
    loss_helper = training_loss.TrainingLossComputer(config.loss, device=device)
    
    global_epoch = 0 
    while True:
        # GRPO Sampling
        for epoch, train_batch in enumerate(train_loader):
            # Sampling Stage
            # Repeat n times for GRPO
            model.eval()
            batch_size = config.batch_size
            
            train_batch = train_batch.expand_sequence(config.group_size)
            expanded_sequences = train_batch.T_world_cpf        
            
            all_log_probs = []
            all_rewards = []
            all_samples = []
            each_rewards = {}
            
            ###for the sake of convenience, we use the same latents for all prompts in a batch.
            global_input_latents = torch.randn(
                    (1, train_batch.T_world_cpf.shape[1], model.get_d_state()),
                    device=accelerator.device,
                )

            for i in tqdm(range(0, len(expanded_sequences), batch_size), desc=f'Sampling Epoch = {global_epoch}'): # 
                current_batch = train_batch.slice_batch(slice(i, i+batch_size))
                
                if i % config.group_size == 0:
                    input_latents = global_input_latents.repeat(batch_size,1,1).clone()
                
                with torch.no_grad():
                    samples, samples_packed, logprobs = run_sampling_with_logprob( # Samples -> all_latents
                        model,
                        body_model=body_model,
                        guidance_mode="no_hands",
                        Ts_world_cpf=current_batch.T_world_cpf,
                        T_cpf_tm1_cpf_t=current_batch.T_cpf_tm1_cpf_t,
                        hamer_detections=None,
                        aria_detections=None,
                        num_samples=1,
                        floor_z=0.0,
                        device=device,
                        guidance_verbose=False,
                        return_packed=True,
                        eta = config.eta,
                        global_input_latent = input_latents
                    )
                
                rewards, reward_dict = compute_rewards(samples, current_batch, body_model, config, reward_model)
                log_probs = torch.stack(logprobs, dim=1).detach() # (4, num_steps, ...)
                samples_packed = torch.stack(samples_packed, dim = 1).detach() # (4, num_steps+1, ...)
                
                all_log_probs.append(log_probs)
                all_samples.append(samples_packed)
                all_rewards.append(rewards)
                
                if len(each_rewards) == 0:
                    each_rewards = reward_dict
                else:
                    for k in reward_dict.keys():
                        each_rewards[k] = torch.cat([each_rewards[k], reward_dict[k]], dim = 0)
                
                torch.cuda.empty_cache()
                    
            all_samples = torch.cat(all_samples, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
            all_rewards = torch.cat(all_rewards, dim=0).to(torch.float32)
            
            timesteps = quadratic_ts(return_tensor=True).to(device=device).repeat(config.batch_size * config.group_size, 1) 
            conditions_T_world_cpf = train_batch.T_world_cpf.to(device).to(torch.float32)
            conditions_T_cpf_tm1_cpf_t = train_batch.T_cpf_tm1_cpf_t.to(device).to(torch.float32)
            
            if epoch % config.save_vis_freq == 0:
                vis_meshes(all_samples, conditions_T_world_cpf, body_model, config.vis_interval, save_path = f'{experiment_dir}/visualization/{epoch}.obj')
            
            samples_dict={
                "conditions_T_world_cpf": conditions_T_world_cpf, # [batch, seq_len, 7]
                "conditions_T_cpf_tm1_cpf_t": conditions_T_cpf_tm1_cpf_t, 
                "timesteps": timesteps[:, :-2], # [batch, 29]
                "packed_traj": all_samples[:, :-1][:, :-1],  # each entry is the latent before timestep t (0, 29) 
                "next_packed_traj": all_samples[:, 1:][:, :-1],  # each entry is the latent after timestep t (1, 30)
                "log_probs": all_log_probs[:, :-1],
                "rewards": all_rewards,
            }
            
            n = len(all_rewards) // (config.group_size)
            advantages = torch.zeros_like(all_rewards)
            
            # compute advantages within each group
            for i in range(n): 
                start_idx = i * config.group_size
                end_idx = (i + 1) * config.group_size
                # Each reward group normalization
                
                group_rewards = all_rewards[start_idx:end_idx]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std # group reward
                
            samples_dict["advantages"] = advantages
            samples_dict["final_advantages"] = advantages
            
            total_batch_size, num_timesteps = samples_dict["timesteps"].shape #256, 29
            
            # model training loop
            model.train()
            for inner_epoch in range(config.num_inner_epochs):
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=accelerator.device)
                        for _ in range(total_batch_size)
                    ]
                )
                for key in ["timesteps", "packed_traj", "next_packed_traj", "log_probs"]:
                    samples_dict[key] = samples_dict[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None],
                        perms,
                    ]

                # # rebatch for training
                samples_batched = {
                    k: v.reshape(-1, config.batch_size, *v.shape[1:])
                    for k, v in samples_dict.items()
                }

                # dict of lists -> list of dicts for easier iteration
                samples_batched = [
                    dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
                ]
                
                for i, sample in tqdm( # For each sample
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    position=0,
                    disable=not accelerator.is_local_main_process,
                ):
                    for j in tqdm(
                        range(num_timesteps),
                        desc=f"Epoch {epoch}.{inner_epoch}: training Timestep",
                        position=1,
                        leave=False
                    ):
                        with accelerator.accumulate(model):
                            sample_timestep = {}
                            for k in sample.keys():
                                if k in ["timesteps", "packed_traj", "next_packed_traj", "log_probs"]:
                                    sample_timestep[k] = sample[k][:, j]
                                else:
                                    sample_timestep[k] = sample[k]
                                
                            # model forward
                            loss = loss_helper.compute_denoising_loss_grpo(
                                model,
                                unwrapped_model=accelerator.unwrap_model(model),
                                sample_batch=sample_timestep,
                                eta = config.eta)
                    
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                            optim.step()
                            optim.zero_grad()
                        
                # Print status update to terminal.
                mem_free, mem_total = torch.cuda.mem_get_info()
                
                if config.enable_reward_model and config.reward_model_path is not None:
                    logger.info(
                        #f"step: {step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                        f" mem: {(mem_total - mem_free) / 1024**3:.2f}/{mem_total / 1024**3:.2f}G"
                        #f" lr: {scheduler.get_last_lr()[0]:.7f}"
                        f" loss: {loss.item():.6f}"
                        f" reward: {all_rewards.mean().item():.6f}"
                        f" foot_skate_reward: {each_rewards['foot_skate_reward'].mean().item():.6f}"
                        f" mpjpe_reward: {each_rewards['mpjpe_reward'].mean().item():.6f}"
                        f" pampjpe_reward: {each_rewards['pampjpe_reward'].mean().item():.6f}"
                        f" gp_reward: {each_rewards['gp_reward'].mean().item():.6f}"
                        f" mpjve_reward: {each_rewards['mpjve_reward'].mean().item():.6f}"
                        f" mpjre_reward: {each_rewards['mpjre_reward'].mean().item():.6f}"
                        f" critic_score: {each_rewards['critic_score'].mean().item():.6f}"
                    )    
                    writer.add_scalar('Loss/train', loss.item(), global_epoch)
                    writer.add_scalar('Reward/reward', all_rewards.mean().item(), global_epoch)
                    writer.add_scalar('Reward/foot_skate_reward', each_rewards['foot_skate_reward'].mean().item(), global_epoch)
                    writer.add_scalar('Reward/mpjpe_reward', each_rewards['mpjpe_reward'].mean().item(), global_epoch)
                    writer.add_scalar('Reward/pampjpe_reward', each_rewards['pampjpe_reward'].mean().item(), global_epoch)
                    writer.add_scalar('Reward/mpjve_reward', each_rewards['mpjve_reward'].mean().item(), global_epoch)
                    writer.add_scalar('Reward/mpjre_reward', each_rewards['mpjre_reward'].mean().item(), global_epoch)
                
                global_epoch += 1

            # Checkpointing.
            if epoch % config.save_ckpt_freq == 0:
                # Save checkpoint.
                checkpoint_path = experiment_dir / f"checkpoints_{epoch}"
                accelerator.save_state(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    tyro.cli(run_training)
