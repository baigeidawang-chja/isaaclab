from pxr import UsdGeom, UsdPhysics, Gf, Usd, PhysxSchema
import omni.usd
import omni.kit.commands
from omni.physx.scripts import deformableUtils, physicsUtils

usd_context = omni.usd.get_context()
stage = usd_context.get_stage()



class HelloWorld:
    def __init__(self):
        self._isaac_assets_path = "/media/chja/CCE54D158C95990271/Assets"
        self.cube_url = self._isaac_assets_path + "/Isaac/4.5/Isaac/Props/Blocks/nvidia_cube.usd"
        self.car_url = "/home/chja/myproject/car.usd"
        self.terrain_path = "/media/chja/CCE54D158C95990271/Assets/Isaac/4.5/Isaac/Environments/Terrains/kkk.usdc"
        self.stage = stage
        self.setup_scene()
    
    def setup_scene(self):

        TerrainXform = UsdGeom.Xform.Define(self.stage, "/World/Terrain")
        omni.kit.commands.execute(
        "IsaacSimSpawnPrim",
        usd_path=self.terrain_path,  
        prim_path="/World/Terrain",
        translation=(0, 0, 0),  
        )
        terrain_mesh = UsdGeom.Mesh.Get(self.stage, "/World/Terrain")
        physicsUtils.set_or_add_translate_op(terrain_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.0))
        physicsUtils.set_or_add_orient_op(terrain_mesh, orient=Gf.Quatf(0, 0, 0, 0))
        physicsUtils.set_or_add_scale_op(terrain_mesh, scale=Gf.Vec3f(0.05, 0.05, 0.05))

        UsdPhysics.RigidBodyAPI.Apply(TerrainXform.GetPrim())
        UsdPhysics.CollisionAPI.Apply(TerrainXform.GetPrim())
        collisionMeshAPI = UsdPhysics.MeshCollisionAPI.Apply(TerrainXform.GetPrim())
        collisionMeshAPI.CreateApproximationAttr(PhysxSchema.Tokens.sdf)
        sdfMeshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(TerrainXform.GetPrim())
        sdfMeshCollision.CreateSdfResolutionAttr(300)

        omni.kit.commands.execute(
        "IsaacSimSpawnPrim",
        usd_path=self.car_url,
        prim_path="/World/Car",
        translation=(0, 0, 2), 
        )
        car_mesh = UsdGeom.Mesh.Get(self.stage, "/World/Car")
        physicsUtils.set_or_add_translate_op(car_mesh, translate=Gf.Vec3f(0.0, 0.0, 2.0))
        physicsUtils.set_or_add_orient_op(car_mesh, orient=Gf.Quatf(0, 0, 0, 0))
        physicsUtils.set_or_add_scale_op(car_mesh, scale=Gf.Vec3f(0.1, 0.1, 0.1))

hello_world = HelloWorld()
