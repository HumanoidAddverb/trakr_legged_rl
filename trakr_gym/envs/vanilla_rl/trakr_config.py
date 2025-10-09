from trakr_gym.envs.base.legged_robot_config import RobotCfg, RobotCfgPPO

class VanillaRLConfig(RobotCfg):
    class env(RobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain(RobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        use_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 2 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [1.0, 0., 0., 0, 0.]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands(RobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(RobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.5, 1.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(RobotCfg.init_state):
        pos = [0.0, 0.0, 0.255] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'LF_adduction': 0.,   # [rad]
            'RF_adduction': 0.0,   # [rad]
            'LB_adduction': 0 ,  # [rad]
            'RB_adduction': 0,   # [rad]

            'LF_hip': 0.0,   # [rad]
            'RF_hip': 0.0,   # [rad]
            'LB_hip': 0 ,  # [rad]
            'RB_hip': 0,   # [rad]

            'LF_knee': 0.0,   # [rad]
            'RF_knee': 0.0,   # [rad]
            'LB_knee': 0.0,  # [rad]
            'RB_knee': 0.0,   # [rad]
        }
    class control(RobotCfg.control):
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'LF_adduction': 20.0,
            'RF_adduction': 20.0,
            'LB_adduction': 20.0,
            'RB_adduction': 20.0,

            'LF_hip': 20.0,
            'RF_hip': 20.0,
            'LB_hip': 20.0,
            'RB_hip': 20.0,

            'LF_knee': 20.0,
            'RF_knee': 20.0,
            'LB_knee': 20.0,
            'RB_knee': 20.0,
        }  # [N*m/rad]
        damping = {'LF_adduction': 2.0,
            'RF_adduction': 2.0,
            'LB_adduction': 2.0,
            'RB_adduction': 2.0,

            'LF_hip': 2.0,
            'RF_hip': 2.0,
            'LB_hip': 2.0,
            'RB_hip': 2.0,

            'LF_knee': 2.0,
            'RF_knee': 2.0,
            'LB_knee': 2.0,
            'RB_knee': 2.0,  }     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(RobotCfg.asset):
        file = '{TRAKR_GYM_ROOT_DIR}/resources/robots/trakr/urdf/robot.urdf'
        name = "trakr"  # actor name
        foot_name = "toe" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["thigh", "hip", "shank"]
        terminate_after_contacts_on = ["hip", "base"]
        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        use_physx_armature = True
        enable_gyroscopic_forces = True
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.1
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.015
        thickness = 0.01
        torque_limit = 30.0
        dof_armature = (0.0097, 0.0097, 0.024)
        dof_friction = 0.

    class domain_rand(RobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.15, 2.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 2.5
        displace_com = False
        com_displacement_range = [-0.15, 0.15]

    class rewards(RobotCfg.rewards):
        class scales(RobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.75
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -5. 
            feet_air_time = 1.0
            collision = -0.5
            stumble = 0. #-0.05 
            action_rate = -0.02
            stand_still = -0.02
            dof_pos_limits = -10.0
            torque_limits = -1.0
            feet_contact_forces = -0.005
            hip = -0.25
            energy = 0. #-0.0002

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.7 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.9
        base_height_target = 0.25
        max_contact_force = 120. # forces above this value are penalized

    class normalization(RobotCfg.normalization):
        class obs_scales(RobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(RobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales(RobotCfg.noise.noise_scales):
            dof_pos = 0.02 # 0.01
            dof_vel = 1.5
            lin_vel = 0.2
            ang_vel = 0.2
            gravity = 0.1 # 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer(RobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(RobotCfg.sim):
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx(RobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.005   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class VanillaRLConfigPPO(RobotCfgPPO):
    seed = 4007
    runner_class_name = 'OnPolicyRunner'
    class policy(RobotCfgPPO.policy):
        init_noise_std = 0.75
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(RobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 0.75
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(RobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'vanilla_rl'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
