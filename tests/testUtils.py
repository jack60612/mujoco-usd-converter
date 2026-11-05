# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
import mujoco
import numpy as np
from pxr import Gf, Tf, Usd, UsdGeom

from mujoco_usd_converter._impl.utils import set_schema_attribute, set_transform
from tests.util.ConverterTestCase import ConverterTestCase


class TestSetSchemaAttribute(ConverterTestCase):

    def test_set_unique_value(self):
        stage: Usd.Stage = Usd.Stage.CreateInMemory()
        prim: Usd.Prim = stage.DefinePrim("/TestPrim")
        prim.ApplyAPI("MjcJointAPI")

        # Setting a valid attribute with a non-default value should author the value
        set_schema_attribute(prim, "mjc:armature", 0.5)
        self.assertTrue(prim.GetAttribute("mjc:armature").HasAuthoredValue())
        self.assertEqual(prim.GetAttribute("mjc:armature").Get(), 0.5)

    def test_set_default_value(self):
        stage: Usd.Stage = Usd.Stage.CreateInMemory()
        prim: Usd.Prim = stage.DefinePrim("/TestPrim")
        prim.ApplyAPI("MjcJointAPI")

        set_schema_attribute(prim, "mjc:armature", 0)  # Default value is 0
        self.assertFalse(prim.GetAttribute("mjc:armature").HasAuthoredValue())  # Should not be authored
        self.assertEqual(prim.GetAttribute("mjc:armature").Get(), 0)  # But should still return default value

    def test_invalid_schema(self):
        stage: Usd.Stage = Usd.Stage.CreateInMemory()
        prim: Usd.Prim = stage.DefinePrim("/TestPrim")
        self.assertFalse(prim.HasAPI("MjcJointAPI"))

        with self.assertRaises(Tf.ErrorException, msg="Attribute mjc:armature is not valid for prim /TestPrim with schemas []"):
            set_schema_attribute(prim, "mjc:armature", 0.5)

    def test_invalid_attribute(self):
        stage: Usd.Stage = Usd.Stage.CreateInMemory()
        prim: Usd.Prim = stage.DefinePrim("/TestPrim")
        prim.ApplyAPI("MjcJointAPI")

        # Setting an invalid attribute should raise a coding error
        with self.assertRaises(Tf.ErrorException):
            set_schema_attribute(prim, "mjc:invalid_attribute", 123)

    def test_invalid_value(self):
        stage: Usd.Stage = Usd.Stage.CreateInMemory()
        prim: Usd.Prim = stage.DefinePrim("/TestPrim")
        prim.ApplyAPI("MjcJointAPI")

        with self.assertRaises(Tf.ErrorException):
            set_schema_attribute(prim, "mjc:armature", "wrong type")


class TestTransformPrecision(ConverterTestCase):

    def test_near_identity_quaternion_is_narrowed_only_at_publication(self):
        # This rotation previously changed its imaginary components by thousands
        # of Float ulps when a Quatf went through Gf.Rotation's axis-angle form.
        quat = np.array([0.9999867878864059, -0.00242414045865189, 0.00258696923225921, -0.00372225547969671])
        quat /= np.linalg.norm(quat)
        for orientation in ("body", "frame", "fromto"):
            with self.subTest(orientation=orientation):
                spec = mujoco.MjSpec()
                stage = Usd.Stage.CreateInMemory()
                prim = UsdGeom.Xform.Define(stage, "/Body")
                if orientation == "frame":
                    frame = spec.worldbody.add_frame(quat=quat)
                    body = frame.add_body()
                    expected = quat
                elif orientation == "fromto":
                    # The same near-identity axis rotation, represented by an
                    # explicit capsule segment rather than an authored quat.
                    body = spec.worldbody.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[0.01, 0, 0],
                        fromto=[0, 0, 0, 0.004848, -0.005174, 1],
                    )
                    axis = np.array(body.fromto[3:]) - np.array(body.fromto[:3])
                    axis /= np.linalg.norm(axis)
                    expected = np.array([1 + axis[2], -axis[1], axis[0], 0])
                    expected /= np.linalg.norm(expected)
                else:
                    body = spec.worldbody.add_body(quat=quat)
                    expected = quat
                set_transform(prim, body, spec)
                (orient,) = (op.Get() for op in prim.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient)
                self.assertIsInstance(orient, Gf.Quatf)
                actual = np.array([orient.GetReal(), *orient.GetImaginary()], dtype=np.float32)
                nonzero = expected != 0
                np.testing.assert_array_equal(actual[nonzero], expected.astype(np.float32)[nonzero])
                # Matrix decomposition can leave cancellation residue on an
                # analytically zero axis. Bound that residue by double arithmetic
                # roundoff scaled by the rotation, not a physical/angular tolerance.
                roundoff = np.finfo(np.float64).eps * np.linalg.norm(expected[1:])
                self.assertTrue(np.all(np.abs(actual[~nonzero]) <= roundoff))

    def test_rotation_equivalence_oracle_preserves_angular_accuracy(self):
        quarter_turn = Gf.Quatf(Gf.Rotation(Gf.Vec3d.ZAxis(), 90).GetQuat())
        self.assertRotationsEquivalent(quarter_turn, -quarter_turn)
        self.assertRotationsEquivalent(Gf.Rotation(Gf.Vec3d.ZAxis(), 90), Gf.Rotation(-Gf.Vec3d.ZAxis(), 270))
        with self.assertRaises(AssertionError):
            self.assertRotationsEquivalent(quarter_turn, Gf.Rotation(Gf.Vec3d.ZAxis(), 90.001))
