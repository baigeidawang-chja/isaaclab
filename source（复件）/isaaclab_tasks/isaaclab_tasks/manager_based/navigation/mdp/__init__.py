# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .pre_trained_policy_action import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403

# Stage-2 distillation modules (uncomment after creating the files)
# from .relevance_map import RelevanceMapGenerator
# from .grid_estimator import GridEstimatorV2
# from .student_distillation_loss import StudentDistillationLoss
