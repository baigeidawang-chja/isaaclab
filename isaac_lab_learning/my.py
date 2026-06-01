from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
import carb
import omni.kit.app


import torch
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
#from isaacsim.examples.interactive.base_sample import BaseSample
#from isaacsim.examples.interactive import BaseSample
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import UsdGeom, Gf, UsdPhysics, Usd, PhysxSchema
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


import omni.usd

print(sys.path)

framework = carb.get_framework()
framework.load_plugins(
    loaded_file_wildcards=["omni.kit.app.plugin"],
    search_paths=[os.path.abspath(f'{os.environ["CARB_APP_PATH"]}/kernel/plugins')],
    )
# Inject a experience config
sys.argv.insert(1, f'{os.environ["EXP_PATH"]}/isaacsim.exp.base.python.kit')

# Add paths to extensions
sys.argv.append(f"--ext-folder")
sys.argv.append(f'{os.path.abspath(os.environ["ISAAC_PATH"])}/exts')

# Run headless
sys.argv.append("--no-window")

enable_extension("omni.kit.widget.stage")
enable_extension("omni.kit.widget.layers")

simulation_app.update()

class HelloWorld:
    def __init__(self):
        self._isaac_assets_path =  "/media/chja/CE54D158C95990271/Assets"
        self.cube_url = self._isaac_assets_path + "/Isaac/4.5/Isaac/Props/Blocks/nvidia_cube.usd"
        self.car_url = "/home/chja/myproject/car.usd"
        self.limo_url = self._isaac_assets_path + "/Isaac/4.5/Isaac/Robots/AgilexRobotics/limo/limo.usd"
        self.terrain_path = "/media/chja/CE54D158C95990271/Assets/Isaac/4.5/Isaac/Environments/Terrains/kkk.usdc"
        self._array_container = torch.Tensor
        self.world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()
        self.num_envs = 10
        self.dimx = 5
        self.dimy = 5
        self.world.scene.add_default_ground_plane()
        self.initial_positions = None
        #self.world = World()
        #self._stage = omni.usd.get_context().get_stage()
        self.setup_scene()
    


    def setup_scene(self):

        #self.world.scene.add_default_ground_plane()
        

        # add_reference_to_stage(usd_path = self.cube_url, prim_path = f"/World/Nvidia_cube")
        # cube_mesh = UsdGeom.Mesh.Get(self.stage, "/World/Nvidia_cube")
        # physicsUtils.set_or_add_translate_op(cube_mesh, translate = Gf.Vec3f(0.0, 0.0, 0.5))
        # physicsUtils.set_or_add_orient_op(cube_mesh, orient = Gf.Quatf(0.92, 0.38, 0 ,0))
        # physicsUtils.set_or_add_scale_op(cube_mesh, scale = Gf.Vec3f(1.2, 1.2, 1.2))

        rootxform = UsdGeom.Xform.Define(self.stage, "/World")
     
        # TerrainXform = UsdGeom.Xform.Define(self.stage,"/World/Terrain")
        # add_reference_to_stage(usd_path = self.terrain_path, prim_path = f"/World/Terrain")
        # terrain_path = "/World/Terrain"
        # terrain_mesh = UsdGeom.Mesh.Get(self.stage, "/World/Terrain")
        # physicsUtils.set_or_add_translate_op(terrain_mesh, translate = Gf.Vec3f(0.0, 0.0, 0.0))
        # physicsUtils.set_or_add_orient_op(terrain_mesh, orient = Gf.Quatf(0, 0, 0 ,0))
        # physicsUtils.set_or_add_scale_op(terrain_mesh, scale = Gf.Vec3f(0.05, 0.05, 0.05))
    
        # add_reference_to_stage(usd_path = self.car_url, prim_path = f"/World/Car")
        # car_mesh = UsdGeom.Mesh.Get(self.stage, "/World/Car")
        # physicsUtils.set_or_add_translate_op(car_mesh, translate = Gf.Vec3f(0.0, 0.0, 2.0))
        # physicsUtils.set_or_add_orient_op(car_mesh, orient = Gf.Quatf(0, 0, 0 ,0))
        # physicsUtils.set_or_add_scale_op(car_mesh, scale = Gf.Vec3f(0.1, 0.1, 0.1))

        # add_reference_to_stage(usd_path = self.limo_url, prim_path = f"/World/Limo")
        # limo_mesh = UsdGeom.Mesh.Get(self.stage, "/World/Limo")
        # physicsUtils.set_or_add_translate_op(limo_mesh, translate = Gf.Vec3f(0.0, 0.0, 5.0))
        # physicsUtils.set_or_add_orient_op(limo_mesh, orient = Gf.Quatf(0, 0, 0 ,0))
        # physicsUtils.set_or_add_scale_op(limo_mesh, scale = Gf.Vec3f(0.1, 0.1, 0.1))

        # terrain_rigid = UsdPhysics.RigidBodyAPI.Apply(self.stage.GetPrimAtPath("/World/Terrain"))
        # terrain_geom = UsdGeom.Cube.Define(self.stage, terrain_path)
        # terrain_prim = terrain_geom.GetPrim()
        # UsdPhysics.RigidBodyAPI.Apply(TerrainXform.GetPrim())
        # UsdPhysics.CollisionAPI.Apply(TerrainXform.GetPrim())
        # UsdPhysics.CollisionAPI.Apply(TerrainXform.GetPrim())
        # UsdPhysics.RigidBodyAPI.Apply(TerrainXform.GetPrim())
        # collisionMeshAPI = UsdPhysics.MeshCollisionAPI.Apply(TerrainXform.GetPrim())
        # collisionMeshAPI.CreateApproximationAttr(PhysxSchema.Tokens.sdf)
        # sdfMeshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(TerrainXform.GetPrim())
        # sdfMeshCollision.CreateSdfResolutionAttr(300)
        
        # jointLocalPositions = [Gf.Vec3f(0.0, 0.0, 0.0),Gf.Vec3f(0.0, 0.0, 0.0)]
        # jointLocalRotations = [Gf.Quatf(1.0), Gf.Quatf(1.0)]
        
        # #给四个车轮以角动量
        # wheel_joint = [
        #    {'name': "RF", "pos": (0.5, 0.3, 0.2)},
        #    {'name': "RR", "pos": (0.5, 0.3, 0.2)},
        #    {'name': "LF", "pos": (0.5, 0.3, 0.2)},
        #    {'name': "LR", "pos": (0.5, 0.3, 0.2)}
        # ]
        
        # for joint_info in wheel_joint:
        #    joint_path = f"/World/Car/{joint_info['name']}_joint"
        #    joint_prim = self.stage.GetPrimAtPath(joint_path)
        #    revolute_joint = UsdPhysics.RevoluteJoint.Get(self.stage,joint_path)
        #    revolute_joint.CreateAxisAttr(UsdPhysics.Tokens.z)
        #    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        #    drive_api.CreateTypeAttr(UsdPhysics.Tokens.force)
        #    drive_api.CreateTargetVelocityAttr(300.0)
        #    drive_api.CreateDampingAttr(200.0)  

    
    
    def run(self):
      while simulation_app.is_running():
        self.world.step(render = True)
        simulation_app.update()

      simulation_app.close()

if __name__ == "__main__":
    hello_world = HelloWorld()
    hello_world.run()
