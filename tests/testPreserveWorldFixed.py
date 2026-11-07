# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
import pathlib

from pxr import Usd, UsdPhysics

import mujoco_usd_converter
from tests.util.ConverterTestCase import ConverterTestCase


class TestPreserveWorldFixed(ConverterTestCase):
    def _convert(self, names=()):
        directory = pathlib.Path(self.tmpDir())
        source = directory / "fixed.xml"
        source.write_text("""<mujoco model="fixed"><compiler inertiafromgeom="false" fusestatic="false"/>
          <worldbody><body name="clamp" pos="1 2 3"><body name="trunk" pos="0 0 1">
            <inertial pos="0 0 0" mass="2" diaginertia=".1 .2 .3"/>
            <geom type="sphere" size=".1"/>
            <body name="tip" pos="1 0 0"><joint name="bend"/>
              <inertial pos="0 0 0" mass="1" diaginertia=".01 .02 .03"/>
              <geom type="sphere" size=".1"/>
            </body>
          </body></body></worldbody></mujoco>""")
        asset = mujoco_usd_converter.Converter(preserve_world_fixed_bodies=names).convert(source, directory / "usd")
        stage = Usd.Stage.Open(asset.path)
        self.assertIsValidUsd(stage)
        return stage

    def test_default_remains_static(self):
        stage = self._convert()
        trunk = stage.GetPrimAtPath("/fixed/Geometry/clamp/trunk")
        self.assertFalse(trunk.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertFalse(trunk.HasAPI(UsdPhysics.MassAPI))

    def test_named_body_retains_mass_and_fixed_joint_not_kinematic_descendants(self):
        stage = self._convert(("trunk",))
        clamp = stage.GetPrimAtPath("/fixed/Geometry/clamp")
        trunk = stage.GetPrimAtPath("/fixed/Geometry/clamp/trunk")
        tip = stage.GetPrimAtPath("/fixed/Geometry/clamp/trunk/tip")
        self.assertFalse(clamp.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertFalse(clamp.HasAPI(UsdPhysics.MassAPI))
        self.assertTrue(trunk.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertEqual(UsdPhysics.MassAPI(trunk).GetMassAttr().Get(), 2.0)
        fixed = [child for child in trunk.GetChildren() if child.IsA(UsdPhysics.FixedJoint)]
        self.assertEqual(len(fixed), 1)
        self.assertEqual(UsdPhysics.Joint(fixed[0]).GetBody1Rel().GetTargets(), [trunk.GetPath()])
        self.assertFalse(UsdPhysics.RigidBodyAPI(trunk).GetKinematicEnabledAttr().Get())
        self.assertFalse(UsdPhysics.RigidBodyAPI(tip).GetKinematicEnabledAttr().Get())
        self.assertTrue(any(child.IsA(UsdPhysics.RevoluteJoint) for child in tip.GetChildren()))

    def test_unknown_body_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown preserved"):
            self._convert(("missing",))
