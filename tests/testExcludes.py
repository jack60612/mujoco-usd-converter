# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import pathlib
from itertools import combinations

import mujoco
import usdex.core
import usdex.test
from pxr import Sdf, Tf, Usd, UsdPhysics

import mujoco_usd_converter
from tests.util.ConverterTestCase import ConverterTestCase


class TestExcludes(ConverterTestCase):

    def test_contact_excludes(self):
        model_path = pathlib.Path("./tests/data/contact_excludes.xml")
        model_name = model_path.stem

        with usdex.test.ScopedDiagnosticChecker(
            self,
            [(Tf.TF_DIAGNOSTIC_WARNING_TYPE, ".*Body 'world' or 'world' not found for contact exclude")],
            level=usdex.core.DiagnosticsLevel.eWarning,
        ):
            asset: Sdf.AssetPath = mujoco_usd_converter.Converter().convert(model_path, self.tmpDir())
        stage = Usd.Stage.Open(asset.path)
        self.assertIsValidUsd(stage)

        # Test that the exclude is applied to the bodies
        body1 = stage.GetPrimAtPath(f"/{model_name}/Geometry/no_collide_A")
        self.assertTrue(body1.IsValid())
        self.assertTrue(body1.HasAPI(UsdPhysics.FilteredPairsAPI))
        filtered_pairs_api = UsdPhysics.FilteredPairsAPI(body1)
        self.assertEqual(len(filtered_pairs_api.GetFilteredPairsRel().GetTargets()), 2)
        self.assertEqual(filtered_pairs_api.GetFilteredPairsRel().GetTargets()[0], f"/{model_name}/Geometry/no_collide_B")
        self.assertEqual(filtered_pairs_api.GetFilteredPairsRel().GetTargets()[1], f"/{model_name}/Geometry/no_collide_C")

        body2 = stage.GetPrimAtPath(f"/{model_name}/Geometry/no_collide_B")
        self.assertTrue(body2.IsValid())
        self.assertTrue(body2.HasAPI(UsdPhysics.FilteredPairsAPI))
        filtered_pairs_api = UsdPhysics.FilteredPairsAPI(body2)
        self.assertEqual(len(filtered_pairs_api.GetFilteredPairsRel().GetTargets()), 1)
        self.assertEqual(filtered_pairs_api.GetFilteredPairsRel().GetTargets()[0], f"/{model_name}/Geometry/no_collide_C")

        # Test that the exclude is not applied to the bodies
        body3 = stage.GetPrimAtPath(f"/{model_name}/Geometry/no_collide_C")
        self.assertTrue(body3.IsValid())
        self.assertFalse(body3.HasAPI(UsdPhysics.FilteredPairsAPI))

    def test_static_body_excludes_target_only_owned_colliders(self):
        model_path = pathlib.Path(self.tmpDir()) / "static_excludes.xml"
        model_path.write_text(
            """<mujoco model="static_excludes">
              <default><geom type="sphere" size="0.1"/></default>
              <worldbody>
                <body name="a">
                  <geom name="a1"/><geom name="a2" pos="0.3 0 0"/>
                  <geom name="visual" contype="0" conaffinity="0"/>
                  <body name="moving_descendant" pos="0 0 1">
                    <joint type="hinge"/><geom name="descendant_geom"/>
                  </body>
                </body>
                <body name="b" pos="1 0 0">
                  <freejoint/><geom name="b1"/><geom name="b2" pos="0.3 0 0"/>
                  <body name="dynamic_child" pos="0 0 1">
                    <joint type="hinge"/><geom name="dynamic_child_geom"/>
                  </body>
                </body>
                <body name="c" pos="2 0 0">
                  <geom name="c1"/><geom name="c2" pos="0.3 0 0"/>
                </body>
                <body name="empty" pos="3 0 0">
                  <body name="empty_child">
                    <joint type="hinge"/><geom name="empty_child_geom"/>
                  </body>
                </body>
              </worldbody>
              <contact>
                <exclude body1="a" body2="b"/>
                <exclude body1="b" body2="c"/>
                <exclude body1="a" body2="c"/>
                <exclude body1="empty" body2="b"/>
                <exclude body1="b" body2="empty"/>
              </contact>
            </mujoco>"""
        )
        model = mujoco.MjModel.from_xml_path(str(model_path))
        asset = mujoco_usd_converter.Converter().convert(model_path, self.tmpDir("converted"))
        stage = Usd.Stage.Open(asset.path)
        self.assertIsValidUsd(stage)
        colliders = {prim.GetPath(): prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)}
        geom_ids = {path: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prim.GetName()) for path, prim in colliders.items()}

        def endpoint_geoms(path):
            prim = stage.GetPrimAtPath(path)
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                return [geom_ids[path]]
            # A rigid body endpoint owns its direct colliders, not nested
            # separately represented bodies, even when those bodies are welded.
            return [geom_id for geom_path, geom_id in geom_ids.items() if geom_path.GetParentPath() == path]

        actual = set()
        filter_count = 0
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                continue
            source_geoms = endpoint_geoms(prim.GetPath())
            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                filter_count += 1
                for first in source_geoms:
                    for second in endpoint_geoms(target):
                        actual.add(frozenset((first, second)))
        excluded_bodies = {frozenset((int(signature) >> 16, int(signature) & 0xFFFF)) for signature in model.exclude_signature}
        expected = {
            frozenset((first, second))
            for first, second in combinations(geom_ids.values(), 2)
            if frozenset((int(model.geom_bodyid[first]), int(model.geom_bodyid[second]))) in excluded_bodies
        }
        self.assertEqual(actual, expected)
        # a->b: two collider/body targets; b->c: two body/collider
        # targets; a->c: four collider/collider targets. Empty adds none.
        self.assertEqual(filter_count, 8)
        root = "/static_excludes/Geometry"
        for name in ("a", "c", "empty"):
            self.assertFalse(stage.GetPrimAtPath(f"{root}/{name}").HasAPI(UsdPhysics.FilteredPairsAPI))
        self.assertTrue(stage.GetPrimAtPath(f"{root}/b").HasAPI(UsdPhysics.FilteredPairsAPI))
        for name in ("a/moving_descendant", "b/dynamic_child", "empty/empty_child"):
            self.assertFalse(stage.GetPrimAtPath(f"{root}/{name}").HasAPI(UsdPhysics.FilteredPairsAPI))
