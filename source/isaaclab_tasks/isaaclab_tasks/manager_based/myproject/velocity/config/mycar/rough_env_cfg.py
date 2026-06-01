# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab_tasks.manager_based.myproject.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from wheeledlab_assets.mushr import MUSHR_CFG
# from base_config import BaseConfig
##
# Pre-defined configs
##

MY_CAR = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/chja/myproject/car.usd",
        #usd_path=f"/home/chja/myproject/leatherback01.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={

            "RevoluteJoint07":0.0,
            "RevoluteJoint14":0.0,
            "RevoluteJoint21":0.0,
            "RevoluteJoint28":0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "drive_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                              "RevoluteJoint07",
                              "RevoluteJoint14",
                              "RevoluteJoint21",
                              "RevoluteJoint28"
                              ],
            # saturation_effort=1000.0,
            effort_limit=10000,
            velocity_limit=4500,
            stiffness=100.0 ,
            damping=50.0,
            friction=0.0,
        ),
    },
)

# class MyCarCfg(BaseConfig):
#     class terrain:
#         # physical terrains flag
#         apply_water_resistance = False
#         apply_water_wave = False
#         apply_deformable = False

#         # physical terrains parameters
#         water_attitude = 0.5
#         water_terrain_random_scale = [0.8, 1.2]
       
#         terrain_combo = "LLM"       # choose from "curriculum", "LLM", "selected" and "NaT" (Not a terrain) for mesh_type not in ['heightfield', 'trimesh'] and terrain_combo == curriculm
#         GenTe_resource_path = "./GenTe/resources/GenTe"
#         prompt_file = GenTe_resource_path + "/prompts/lang2terrain.txt"
#         function_prompt = GenTe_resource_path + "/prompts/function.json"
#         image_prompt = GenTe_resource_path + "/prompts/image2lang.txt"
#         llm_model_name = GenTe_resource_path + "/models/Llama-3.1-8B-Instruct"
#         vlm_model_load_terrain_dirame = GenTe_resource_path + "/models/Qwen2-VL-7B-Instruct"
#         instruct_lang = "Question: Generate a terrain that depicts a mountainous landscape with rolling hills covered in dense green vegetation. The terrain is characterized by gentle slopes and a variety of elevations."
#         instruct_image_path = GenTe_resource_path + "/images/cityroad.jpeg"

#         load_terrain_dir = "/home/webdrag0n/webdrag0n-storage/haron/llm_bipedwalk/llm_part/mid_result/v7/buildingfield"
 
    
#         mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
#         horizontal_scale = 0.1 # [m]
#         vertical_scale = 0.005 # [m]
#         border_size = 25 # [m]
        
#         static_friction = 1.0
#         dynamic_friction = 1.0
#         restitution = 0.
#         # rough terrain only:
#         measure_heights = True
#         measured_points_x = np.linspace(-0.8,0.8,17).tolist()
#         measured_points_y = np.linspace(-0.5,0.5,11).tolist()
        
#         terrain_kwargs = None # Dict of arguments for selected terrain
#         max_init_terrain_level = 5 # starting curriculum state
#         terrain_length = 20.
#         terrain_width = 20.
#         num_rows= 10 # number of terrain rows (levels)
#         num_cols = 10 # number of terrain cols (types)
#         # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
#         # terrain_proportions = [0.1, 0.1, 0.3, 0.2, 0.15, 0.05, 0.05, 0.05]
#         terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
#         # trimesh only:
#         slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces


@configclass
class MyCarRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = MY_CAR.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Chassis"
        # event
        self.events.push_robot = None
        # self.events.add_base_mass.params["mass_distribution_params"] = (100.0, 100.0)
        # self.events.add_base_mass.params["asset_cfg"].body_names = "Chassis"
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = "Chassis"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-1, 1), "y": (-1, 1),"z": (0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0, 0),
                "y": (1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-1.0, 1.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class MyCarRoughEnvCfg_PLAY(MyCarRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 20
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
